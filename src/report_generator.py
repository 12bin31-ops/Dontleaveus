"""Automated HTML and PDF report generation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import base64
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template


class ReportGenerator:
    """Generate automated HTML/PDF reports."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.report_dir = Path(config['report_generation']['output_dir'])
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.html_template = self._load_template()
    
    def _load_template(self) -> str:
        """Load HTML template."""
        template_path = Path("templates/report_template.html")
        
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return self._get_default_template()
    
    def _get_default_template(self) -> str:
        """Get default HTML template."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f4f4f4;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        
        header {
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .report-info {
            font-size: 0.9em;
            color: #666;
        }
        
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.8em;
        }
        
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .metric-card h3 {
            font-size: 0.9em;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        
        .metric-card .value {
            font-size: 2.5em;
            font-weight: bold;
        }
        
        .metric-card.success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        
        .metric-card.warning {
            background: linear-gradient(135deg, #eb6b56 0%, #f7b801 100%);
        }
        
        .metric-card.danger {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background: #34495e;
            color: white;
            font-weight: bold;
        }
        
        tr:hover {
            background: #f5f5f5;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        .insight-box {
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 3px;
        }
        
        .insight-box h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .recommendation {
            background: #d5f4e6;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 15px 0;
            border-radius: 3px;
        }
        
        .recommendation h4 {
            color: #27ae60;
            margin-bottom: 10px;
        }
        
        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        
        .page-break {
            page-break-after: always;
        }
        
        ul {
            margin-left: 20px;
            margin-top: 10px;
        }
        
        li {
            margin-bottom: 8px;
        }
        
        .risk-high {
            color: #c0392b;
            font-weight: bold;
        }
        
        .risk-medium {
            color: #e67e22;
            font-weight: bold;
        }
        
        .risk-low {
            color: #27ae60;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔮 Customer Churn Prediction Report</h1>
            <div class="report-info">
                <p>Generated: {{ generated_at }}</p>
                <p>Analysis Period: {{ period }}</p>
            </div>
        </header>
        
        <!-- Executive Summary -->
        <section class="section">
            <h2>📋 Executive Summary</h2>
            {{ executive_summary | safe }}
        </section>
        
        <!-- Key Metrics -->
        <section class="section">
            <h2>📊 Model Performance Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card success">
                    <h3>Accuracy</h3>
                    <div class="value">{{ metrics.accuracy | round(2) }}%</div>
                </div>
                <div class="metric-card success">
                    <h3>Precision</h3>
                    <div class="value">{{ metrics.precision | round(2) }}%</div>
                </div>
                <div class="metric-card success">
                    <h3>Recall</h3>
                    <div class="value">{{ metrics.recall | round(2) }}%</div>
                </div>
                <div class="metric-card success">
                    <h3>F1 Score</h3>
                    <div class="value">{{ metrics.f1 | round(2) }}</div>
                </div>
                <div class="metric-card success">
                    <h3>ROC-AUC</h3>
                    <div class="value">{{ metrics.roc_auc | round(2) }}</div>
                </div>
            </div>
        </section>
        
        <!-- Feature Importance -->
        <section class="section">
            <h2>🎯 Top Features Driving Churn</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance Score</th>
                    </tr>
                </thead>
                <tbody>
                    {% for idx, (feature, importance) in enumerate(top_features, 1) %}
                    <tr>
                        <td>{{ idx }}</td>
                        <td>{{ feature }}</td>
                        <td>{{ (importance * 100) | round(2) }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        
        <!-- High Risk Customers -->
        <section class="section">
            <h2>⚠️ High-Risk Customers (Top 10)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Customer ID</th>
                        <th>Churn Risk</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
                    {% for idx, (cid, risk) in enumerate(high_risk_customers, 1) %}
                    <tr>
                        <td>{{ idx }}</td>
                        <td>{{ cid }}</td>
                        <td>{{ (risk * 100) | round(1) }}%</td>
                        <td>
                            {% if risk > 0.7 %}
                                <span class="risk-high">🔴 HIGH</span>
                            {% elif risk > 0.4 %}
                                <span class="risk-medium">🟡 MEDIUM</span>
                            {% else %}
                                <span class="risk-low">🟢 LOW</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        
        <!-- LLM Insights (Harness Engineering) -->
        {% if insights %}
        <section class="section">
            <h2>🤖 AI-Generated Insights (Advanced Analysis)</h2>
            
            {% if insights.summary %}
            <div class="insight-box">
                <h4>Summary Insights</h4>
                {{ insights.summary | safe }}
            </div>
            {% endif %}
            
            {% if insights.feature_analysis %}
            <div class="insight-box">
                <h4>Feature Analysis</h4>
                {{ insights.feature_analysis | safe }}
            </div>
            {% endif %}
            
            {% if insights.risk_segments %}
            <div class="insight-box">
                <h4>Risk Segment Profiles</h4>
                {{ insights.risk_segments | safe }}
            </div>
            {% endif %}
            
            {% if insights.key_takeaways %}
            <div class="insight-box">
                <h4>Key Takeaways (Compressed)</h4>
                {{ insights.key_takeaways | safe }}
            </div>
            {% endif %}
        </section>
        {% endif %}
        
        <!-- Recommendations -->
        {% if recommendations %}
        <section class="section">
            <h2>💡 Recommended Actions</h2>
            {% for rec in recommendations %}
            <div class="recommendation">
                <h4>{{ rec.title }}</h4>
                <p>{{ rec.description }}</p>
                <p><strong>Priority:</strong> {{ rec.priority }}</p>
            </div>
            {% endfor %}
        </section>
        {% endif %}
        
        <!-- Charts -->
        {% if charts %}
        <div class="page-break"></div>
        <section class="section">
            <h2>📈 Visualizations</h2>
            {% for chart_name, chart_base64 in charts.items() %}
            <div class="chart-container">
                <h3>{{ chart_name }}</h3>
                <img src="data:image/png;base64,{{ chart_base64 }}" alt="{{ chart_name }}">
            </div>
            {% endfor %}
        </section>
        {% endif %}
        
        <footer>
            <p>This report was automatically generated by Customer Churn Prediction System</p>
            <p>© {{ current_year }} | For internal use only</p>
        </footer>
    </div>
</body>
</html>
"""
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return image_base64
    
    def generate_report(self, 
                       metrics: Dict[str, float],
                       feature_importance: pd.DataFrame,
                       high_risk_customers: List[tuple],
                       insights: Dict[str, str] = None,
                       recommendations: List[Dict] = None,
                       charts: Dict[str, plt.Figure] = None,
                       executive_summary: str = None) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            metrics: Model performance metrics
            feature_importance: Top features DataFrame
            high_risk_customers: List of (customer_id, churn_probability) tuples
            insights: LLM-generated insights
            recommendations: List of recommendation dicts
            charts: Dict of matplotlib figures
            executive_summary: Custom executive summary text
        
        Returns:
            Path to generated HTML file
        """
        print("\n" + "="*50)
        print("📝 GENERATING AUTOMATED REPORT")
        print("="*50)
        
        # Convert metrics to percentages
        metrics_display = {k: v * 100 for k, v in metrics.items()}
        
        # Convert figures to base64
        charts_base64 = {}
        if charts:
            for chart_name, fig in charts.items():
                charts_base64[chart_name] = self._fig_to_base64(fig)
        
        # Prepare top features
        top_features = list(zip(
            feature_importance['feature'].head(10),
            feature_importance['importance'].head(10)
        ))
        
        # Render template
        template = Template(self.html_template)
        html_content = template.render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            period="Latest Analysis",
            metrics=metrics_display,
            top_features=top_features,
            high_risk_customers=high_risk_customers[:10],  # Top 10
            insights=insights or {},
            recommendations=recommendations or [],
            charts=charts_base64,
            executive_summary=executable_summary or "<p>See detailed metrics and insights below.</p>",
            current_year=datetime.now().year
        )
        
        # Save HTML file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.report_dir / f"churn_report_{timestamp}.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report saved to {html_path}")
        
        # Generate PDF (optional)
        try:
            self._generate_pdf(html_path)
        except Exception as e:
            print(f"⚠️ PDF generation skipped: {str(e)}")
        
        return str(html_path)
    
    def _generate_pdf(self, html_path: str) -> None:
        """Generate PDF from HTML."""
        try:
            from weasyprint import HTML
            
            pdf_path = str(html_path).replace('.html', '.pdf')
            HTML(string=open(html_path).read()).write_pdf(pdf_path)
            print(f"✅ PDF report saved to {pdf_path}")
            
        except ImportError:
            print("⚠️ WeasyPrint not installed - skipping PDF generation")
        except Exception as e:
            print(f"⚠️ PDF generation failed: {str(e)}")
    
    def generate_json_report(self, report_data: Dict[str, Any]) -> str:
        """Generate JSON report for programmatic access."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.report_dir / f"churn_report_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"✅ JSON report saved to {json_path}")
        return str(json_path)
