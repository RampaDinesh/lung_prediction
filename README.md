## 🫁 Built an AI model to detect Pneumonia from chest X-rays — with explainability using Grad-CAM 🔥

Over the past few days, I worked on developing a deep learning model that not only predicts pneumonia but also highlights the regions in the X-ray that influenced the decision.

## What I implemented:
• Started with a custom CNN and improved performance using ResNet18
• Handled class imbalance in the dataset
• Tuned decision threshold to prioritize recall (important in medical tasks)
• Evaluated using precision, recall, F1-score, and confusion matrix
• Integrated Grad-CAM to visualize model focus areas

# 📊 Key insight:
Instead of relying only on accuracy, I focused on improving recall to ensure the model does not miss disease cases. Grad-CAM helps make the model’s predictions more interpretable by showing *where* it is looking.

# ⚠️ Note:
This is an experimental project and not a medical-grade system, but it demonstrates how deep learning + explainability can be applied in healthcare scenarios.

I’m currently working on improving the model further and building a web app for real-time predictions.

Would love to hear your feedback and suggestions! 


