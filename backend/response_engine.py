from datetime import datetime,timezone


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BLOCK_THRESHOLD = 0.90


# --------------------------------------------------
# In-memory security state
# --------------------------------------------------

blocked_sources = set()
quarantined_hosts = set()

alerts = []
live_events = []


# --------------------------------------------------
# Network threat processing
# --------------------------------------------------

def process_threat(source_ip, attack_type, confidence):
    """
    Process one network-security prediction.

    BENIGN traffic is allowed.
    High-confidence malicious traffic is blocked
    at the application level.
    Suspicious traffic generates an alert.
    """

    confidence = float(confidence)

    timestamp = datetime.now(timezone.utc).isoformat()

    # ----------------------------------------------
    # BENIGN
    # ----------------------------------------------

    if str(attack_type).upper() == "BENIGN":

        event = {
            "source_ip": str(source_ip),
            "attack_type": str(attack_type),
            "confidence": round(confidence, 4),
            "severity": "LOW",
            "action": "ALLOWED",
            "timestamp": timestamp
        }

        live_events.append(event)

        return event

    # ----------------------------------------------
    # High-confidence malicious traffic
    # ----------------------------------------------

    if confidence >= BLOCK_THRESHOLD:

        blocked_sources.add(str(source_ip))

        event = {
            "source_ip": str(source_ip),
            "attack_type": str(attack_type),
            "confidence": round(confidence, 4),
            "severity": "HIGH",
            "action": "BLOCKED",
            "timestamp": timestamp
        }

        alerts.append(event)
        live_events.append(event)

        return event

    # ----------------------------------------------
    # Suspicious but below threshold
    # ----------------------------------------------

    event = {
        "source_ip": str(source_ip),
        "attack_type": str(attack_type),
        "confidence": round(confidence, 4),
        "severity": "MEDIUM",
        "action": "ALERT",
        "timestamp": timestamp
    }

    alerts.append(event)
    live_events.append(event)

    return event


# --------------------------------------------------
# Network blocklist
# --------------------------------------------------

def get_blocked_sources():
    return sorted(blocked_sources)


def block_source(source_ip):
    blocked_sources.add(str(source_ip))


def unblock_source(source_ip):
    blocked_sources.discard(str(source_ip))


# --------------------------------------------------
# Alerts
# --------------------------------------------------

def get_alerts():
    return alerts


# --------------------------------------------------
# Live events
# --------------------------------------------------

def get_live_events():
    return live_events


# --------------------------------------------------
# Ransomware host quarantine
# --------------------------------------------------

def quarantine_host(host_ip, reason="Suspicious ransomware behaviour"):

    host_ip = str(host_ip).strip()

    quarantined_hosts.add(host_ip)

    alert = {
        "host": host_ip,
        "reason": reason,
        "action": "QUARANTINED",
        "severity": "CRITICAL",
        "timestamp": datetime.now().isoformat()
    }

    alerts.append(alert)

    return alert
def quarantine_host(
    host_ip,
    reason="Suspicious ransomware behaviour"
):

    host_ip = str(host_ip).strip()

    quarantined_hosts.add(host_ip)

    alert = {
        "host": host_ip,
        "reason": reason,
        "action": "QUARANTINED",
        "severity": "CRITICAL",
        "timestamp": datetime.now().isoformat()
    }

    alerts.append(alert)

    return alert


# --------------------------------------------------
# Automatic Ransomware Behaviour Response
# --------------------------------------------------

def process_ransomware_event(
    host_ip,
    risk_level,
    process_name,
    entropy_score
):
    """
    Process a ransomware behaviour event.

    HIGH risk -> automatic application-level quarantine
    MEDIUM risk -> alert for investigation
    LOW risk -> monitoring only
    """

    host_ip = str(host_ip).strip()
    risk_level = str(risk_level).upper()
    process_name = str(process_name)
    entropy_score = float(entropy_score)

    timestamp = datetime.now().isoformat()

    # --------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------

    if risk_level == "HIGH":

        quarantined_hosts.add(host_ip)

        event = {
            "host": host_ip,
            "process_name": process_name,
            "entropy_score": entropy_score,
            "risk_level": "HIGH",
            "action": "QUARANTINED",
            "severity": "CRITICAL",
            "timestamp": timestamp
        }

        alerts.append(event)

        return event

    # --------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------

    if risk_level == "MEDIUM":

        event = {
            "host": host_ip,
            "process_name": process_name,
            "entropy_score": entropy_score,
            "risk_level": "MEDIUM",
            "action": "ALERT",
            "severity": "HIGH",
            "timestamp": timestamp
        }

        alerts.append(event)

        return event

    # --------------------------------------------------
    # LOW RISK
    # --------------------------------------------------

    event = {
        "host": host_ip,
        "process_name": process_name,
        "entropy_score": entropy_score,
        "risk_level": "LOW",
        "action": "MONITORING",
        "severity": "LOW",
        "timestamp": timestamp
    }

    live_events.append(event)

    return event

def unquarantine_host(host_ip):

    host_ip = str(host_ip).strip()

    was_quarantined = host_ip in quarantined_hosts

    quarantined_hosts.discard(host_ip)

    return {
        "host": host_ip,
        "action": "UNQUARANTINED",
        "was_quarantined": was_quarantined,
        "remaining_quarantined_hosts": sorted(quarantined_hosts),
        "timestamp": datetime.now().isoformat()
    }


def get_quarantined_hosts():

    return sorted(quarantined_hosts)
# --------------------------------------------------
# Potential Unknown / Anomalous Threat Response
# --------------------------------------------------

def process_anomaly_event(
    source_ip,
    anomaly_score
):
    """
    Handle traffic that does not confidently match
    the learned normal baseline.

    This does not automatically block the source.
    It creates an alert for investigation.
    """

    source_ip = str(source_ip).strip()
    anomaly_score = float(anomaly_score)

    timestamp = datetime.now().isoformat()

    event = {
        "source_ip": source_ip,
        "attack_type": "Potential Unknown Threat",
        "anomaly_score": round(anomaly_score, 6),
        "severity": "HIGH",
        "action": "ALERT",
        "timestamp": timestamp
    }

    alerts.append(event)
    live_events.append(event)

    return event