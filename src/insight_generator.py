"""
LLM-based insight generation using Harness Engineering techniques.

This module leverages Claude API with advanced prompting techniques:
- Chain of Thought (CoT): Step-by-step reasoning
- Self-Consistency: Multiple perspectives, select best
- ReAct: Reason-Act-Observe cycle
- Memory Compaction: Compress key findings for efficiency
"""

import json
import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np

try:
    from anthropic import Anthropic
except ImportError:
    print("⚠️ Anthropic SDK not installed. Install with: pip install anthropic")
    Anthropic = None


class InsightGenerator:
    """Generate business insights using LLM with Harness Engineering."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_config = config.get('llm', {})
        
        if Anthropic is None:
            print("⚠️ LLM insights disabled - Anthropic SDK not available")
            self.client = None
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("⚠️ ANTHROPIC_API_KEY not set - LLM insights disabled")
                self.client = None
            else:
                self.client = Anthropic()
    
    def _call_claude(self, system_prompt: str, user_message: str) -> str:
        """Call Claude API with given prompts."""
        if self.client is None:
            return "⚠️ LLM not available"
        
        try:
            message = self.client.messages.create(
                model=self.llm_config.get('model', 'claude-opus-4-6'),
                max_tokens=self.llm_config.get('max_tokens', 2000),
                temperature=self.llm_config.get('temperature', 0.7),
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            print(f"❌ Claude API error: {str(e)}")
            return f"Error calling Claude: {str(e)}"
    
    # ==================== HARNESS ENGINEERING TECHNIQUES ====================
    
    def chain_of_thought(self, context: str, question: str) -> str:
        """
        Chain of Thought (CoT): Encourage step-by-step reasoning.
        
        Let the model think through the problem systematically.
        """
        system_prompt = """You are a data science expert analyzing customer churn patterns.
Use step-by-step reasoning to answer questions.
Break down complex analysis into logical steps.
Show your reasoning clearly."""
        
        user_message = f"""Context:
{context}

Question: {question}

Please answer by thinking through this step-by-step:
1. First, identify the key patterns
2. Then, analyze what they mean
3. Finally, draw conclusions"""
        
        return self._call_claude(system_prompt, user_message)
    
    def self_consistency(self, context: str, question: str) -> str:
        """
        Self-Consistency: Generate multiple perspectives and select best.
        
        Get different viewpoints and synthesize them.
        """
        perspectives = []
        
        # Perspective 1: Business Impact
        system_1 = """Analyze this churn data from a business impact perspective.
Focus on revenue implications and customer retention strategies."""
        
        # Perspective 2: Technical Pattern Analysis
        system_2 = """Analyze this churn data from a technical data pattern perspective.
Focus on statistical significance and feature correlations."""
        
        # Perspective 3: Customer Segment Analysis
        system_3 = """Analyze this churn data from a customer segment perspective.
Focus on specific customer groups and their behavior patterns."""
        
        for system_prompt in [system_1, system_2, system_3]:
            perspective = self._call_claude(
                system_prompt, 
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            perspectives.append(perspective)
        
        # Synthesize perspectives
        synthesis_prompt = f"""Given these three different perspectives on the same data:

PERSPECTIVE 1 (Business Impact):
{perspectives[0]}

PERSPECTIVE 2 (Technical Pattern):
{perspectives[1]}

PERSPECTIVE 3 (Customer Segment):
{perspectives[2]}

Please synthesize these into a comprehensive insight that combines the best elements of each perspective."""
        
        synthesis = self._call_claude(
            "You are synthesizing different analytical perspectives into unified insights.",
            synthesis_prompt
        )
        
        return synthesis
    
    def react_reasoning(self, context: str, question: str) -> str:
        """
        ReAct (Reason-Act-Observe): Iterative cycle of reasoning, action, observation.
        
        Think about what to do, do it, observe results, repeat.
        """
        system_prompt = """You are a churn analysis expert using the ReAct framework.
Follow this cycle:
1. REASON: What should we analyze next?
2. ACT: Perform the analysis mentally
3. OBSERVE: What patterns emerge?
4. REPEAT: Iterate to find deeper insights

Format your response clearly with these sections."""
        
        user_message = f"""Context:
{context}

Question: {question}

Use the ReAct framework to provide comprehensive analysis."""
        
        return self._call_claude(system_prompt, user_message)
    
    def memory_compaction(self, findings: List[str]) -> str:
        """
        Memory Compaction: Compress findings into key takeaways.
        
        Distill long findings into concise, actionable insights.
        """
        findings_text = "\n".join([f"- {f}" for f in findings])
        
        system_prompt = """You are compressing detailed findings into concise key insights.
Each insight should be:
1. Specific and actionable
2. Business-relevant
3. Evidence-based
4. Concise (1-2 sentences max)"""
        
        user_message = f"""Here are detailed findings from churn analysis:

{findings_text}

Please compress these into 3-5 key takeaways that are:
- Specific and actionable
- Business-relevant  
- Ready for executive summary"""
        
        return self._call_claude(system_prompt, user_message)
    
    # ==================== INSIGHT GENERATION METHODS ====================
    
    def generate_summary_insight(self, metrics: Dict[str, float], 
                                feature_importance: pd.DataFrame) -> str:
        """Generate high-level summary insights."""
        if self.client is None:
            return "LLM insights not available"
        
        context = f"""Model Performance Metrics:
- Accuracy: {metrics.get('accuracy', 0):.4f}
- Precision: {metrics.get('precision', 0):.4f}
- Recall: {metrics.get('recall', 0):.4f}
- F1 Score: {metrics.get('f1', 0):.4f}
- ROC-AUC: {metrics.get('roc_auc', 0):.4f}

Top Important Features:
{feature_importance.head(10).to_string()}
"""
        
        question = "What are the key business insights about customer churn patterns?"
        
        # Use Self-Consistency for comprehensive summary
        return self.self_consistency(context, question)
    
    def generate_feature_analysis(self, feature_importance: pd.DataFrame) -> str:
        """Generate insights about important features."""
        if self.client is None:
            return "LLM insights not available"
        
        context = f"""Top Features Driving Churn:
{feature_importance.head(15).to_string()}

These features represent different aspects of customer relationship:
- Tenure: How long customer has been with company
- Contract: Type of contract (month-to-month, 1-year, 2-year)
- Internet Service: Type of service used
- Charges: Monthly and total charges
"""
        
        question = "Why are these features important for churn? What do they tell us about risky customers?"
        
        # Use Chain of Thought for detailed explanation
        return self.chain_of_thought(context, question)
    
    def generate_risk_segment_insight(self, high_risk_customers: pd.DataFrame,
                                     feature_names: List[str]) -> str:
        """Generate insights about high-risk customer segments."""
        if self.client is None:
            return "LLM insights not available"
        
        # Create summary statistics of high-risk group
        risk_summary = high_risk_customers[feature_names].describe().to_string()
        
        context = f"""High-Risk Customer Characteristics (Top 10% Churn Probability):

{risk_summary}

Number of high-risk customers: {len(high_risk_customers)}
Average churn probability: {high_risk_customers.get('churn_probability', pd.Series([0])).mean():.2%}
"""
        
        question = "What is the profile of high-risk customers? What interventions would be most effective?"
        
        # Use ReAct for iterative deep analysis
        return self.react_reasoning(context, question)
    
    def generate_actionable_recommendations(self, insights: List[str]) -> str:
        """Generate actionable business recommendations."""
        if self.client is None:
            return "Recommendations not available"
        
        system_prompt = """You are a business consultant translating data insights into action.
For each insight, provide:
1. Specific action to take
2. Target customer segment
3. Expected impact
4. Implementation priority (High/Medium/Low)

Format as a structured action plan."""
        
        insights_text = "\n".join([f"- {i}" for i in insights])
        
        user_message = f"""Based on these churn analysis insights:

{insights_text}

Create an action plan with specific, implementable recommendations for reducing customer churn."""
        
        return self._call_claude(system_prompt, user_message)
    
    def generate_executive_summary(self, metrics: Dict[str, float],
                                   key_findings: List[str],
                                   recommendations: List[str]) -> str:
        """Generate executive summary combining all insights."""
        if self.client is None:
            return "Summary not available"
        
        findings_text = "\n".join([f"- {f}" for f in key_findings])
        recs_text = "\n".join([f"- {r}" for r in recommendations])
        
        system_prompt = """You are writing an executive summary for a churn prediction analysis.
The summary should:
1. Be concise (3-4 paragraphs max)
2. Focus on business impact
3. Emphasize key insights and recommendations
4. Be suitable for C-level executives"""
        
        user_message = f"""Create an executive summary with:

Model Performance:
- Accuracy: {metrics.get('accuracy', 0):.2%}
- Precision: {metrics.get('precision', 0):.2%}
- Recall: {metrics.get('recall', 0):.2%}

Key Findings:
{findings_text}

Recommended Actions:
{recs_text}"""
        
        return self._call_claude(system_prompt, user_message)
    
    def batch_generate_insights(self, metrics: Dict[str, float],
                               feature_importance: pd.DataFrame,
                               high_risk_customers: pd.DataFrame = None,
                               feature_names: List[str] = None) -> Dict[str, str]:
        """Generate all insights in batch."""
        print("\n" + "="*50)
        print("🤖 GENERATING LLM-BASED INSIGHTS (Harness Engineering)")
        print("="*50)
        
        insights = {}
        
        if not self.config['llm'].get('harness_engineering', {}).get('enabled', True):
            print("⚠️ Harness Engineering disabled in config")
            return insights
        
        print("\n1️⃣ Generating summary insights (Self-Consistency)...")
        try:
            insights['summary'] = self.generate_summary_insight(metrics, feature_importance)
            print("✅ Summary insights generated")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("\n2️⃣ Generating feature analysis (Chain of Thought)...")
        try:
            insights['feature_analysis'] = self.generate_feature_analysis(feature_importance)
            print("✅ Feature analysis generated")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        if high_risk_customers is not None and feature_names is not None:
            print("\n3️⃣ Generating risk segment insights (ReAct)...")
            try:
                insights['risk_segments'] = self.generate_risk_segment_insight(
                    high_risk_customers, feature_names
                )
                print("✅ Risk segment insights generated")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print("\n4️⃣ Compressing findings (Memory Compaction)...")
        try:
            key_findings = list(insights.values())[:3]  # Take first 3 insights
            insights['key_takeaways'] = self.memory_compaction(key_findings)
            print("✅ Key takeaways extracted")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        return insights
