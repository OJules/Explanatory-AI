# 📊 Results Directory - Explanatory AI Analysis

This directory contains interpretability analysis results, method comparisons, and XAI visualizations from the comprehensive explainable AI study.

## 📁 Contents

### **SHAP Analysis Results**
- `shap_values_tabular.csv` - Feature importance scores for tabular data
- `shap_summary_plots.png` - Global feature importance visualizations
- `shap_waterfall_charts.png` - Individual prediction explanations
- `shap_interaction_effects.png` - Feature interaction analysis

### **Computer Vision Explanations**
- `gradcam_heatmaps/` - Grad-CAM visualization outputs
- `lrp_attribution_maps.png` - Layer-wise relevance propagation results
- `integrated_gradients_vision.png` - Path attribution visualizations
- `occlusion_sensitivity_maps.png` - Input perturbation analysis

### **NLP Interpretability**
- `attention_visualizations.png` - Transformer attention patterns
- `bert_token_attributions.csv` - Word-level importance scores
- `counterfactual_examples.json` - Text modification explanations
- `lime_text_explanations.png` - Local text interpretability

### **Graph Neural Network Explanations**
- `gnn_subgraph_explanations.png` - Important graph structures
- `node_attribution_scores.csv` - Node importance measurements
- `molecular_property_explanations.png` - Chemical structure insights

### **Comparative Analysis**
- `method_comparison_metrics.csv` - Quantitative XAI method evaluation
- `faithfulness_scores.json` - Explanation reliability measurements
- `computational_efficiency.csv` - Runtime and resource usage
- `human_evaluation_results.pdf` - User study outcomes

### **Cross-Domain Analysis**
- `xai_method_applicability.csv` - Method suitability by data type
- `consistency_analysis.json` - Explanation stability metrics
- `bias_detection_results.png` - Fairness through interpretability

## 🎯 How to Generate Results

Run the XAI analysis scripts to populate this directory:
```bash
python xai_comparative_analysis.py
python xai_advanced_methods.py
```

The comprehensive workflow:
1. **Data Preparation** - Multi-modal dataset preprocessing
2. **Model Training** - Domain-specific ML model development
3. **XAI Application** - Method implementation across domains
4. **Evaluation** - Quantitative and qualitative assessment

## 📈 Key Interpretability Indicators

Expected analysis outputs:
- **Feature Importance Rankings** - Global and local explanations
- **Attribution Quality Scores** - Explanation faithfulness metrics
- **Method Applicability** - Domain-specific performance
- **Computational Efficiency** - Practical deployment considerations

---

*Note: Results demonstrate the comparative effectiveness of interpretability methods across different AI applications.*
