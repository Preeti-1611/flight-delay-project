# SkyCast Analytics  - HTML Templates

HERO_SECTION = """
<div class="hero-container">
    <h1 class="hero-title">✈️ SkyCast Analytics</h1>
    <p class="hero-subtitle">Weather-Integrated Flight Delay Forecasting</p>
</div>
"""

def get_prediction_reasons(condition, wind_speed, visibility):
    reasons = []
    if condition == "Storm": 
        reasons.append("Severe weather conditions detected at origin.")
    if wind_speed > 25: 
        reasons.append(f"High wind speeds ({wind_speed} knots) may affect ground operations.")
    if visibility < 2: 
        reasons.append("Low visibility requires increased separation between flights.")
    if not reasons: 
        reasons.append("Factors are based on historical carrier performance and scheduling.")
    
    html = "<h3>Why this prediction?</h3><ul>"
    for r in reasons:
        html += f"<li>{r}</li>"
    html += "</ul>"
    return html
