# FocusMonitor: A Context-Aware Multimodal Framework for Real-Time Study Distraction Detection

Maintaining concentration during computer-based learning is increasingly challenging due to constant digital distractions such as social media, messaging applications, entertainment platforms, and frequent task switching. Most existing distraction detection systems rely on a single source of information—typically facial cues or gaze estimation—which makes them less reliable for identifying real-world study behavior.

**FocusMonitor** is a context-aware multimodal AI framework designed to monitor students' study engagement in real time using only a standard laptop. Instead of depending on a single modality, the system combines computer vision, user interaction, application context, audio context, and surrounding object detection to obtain a comprehensive understanding of a learner's study state.

The framework analyzes facial attention, eye closure, head orientation, keyboard and mouse activity, active application semantics, speech context, and nearby objects. These heterogeneous signals are synchronized and fused using an XGBoost-based decision model to classify four study states: **Focused**, **Distracted**, **Neutral**, and **Absent**. To improve reliability in real-world scenarios, FocusMonitor incorporates context-aware decision rules, temporal smoothing, persistent distraction alerts, and an interactive Streamlit dashboard for real-time feedback and session analytics.

Unlike conventional vision-only approaches, FocusMonitor integrates digital activity with physical behavior, allowing it to distinguish situations where a student appears attentive while using distracting applications, or appears inactive while genuinely studying. The framework operates on standard laptop hardware without requiring wearable devices or specialized sensors, making it suitable for intelligent learning environments, online education, and AI-assisted self-study support.

**Key Features**
• Context-aware multimodal fusion
• Real-time distraction monitoring
• Four-state attention classification
• Computer vision using OpenCV + MediaPipe
• Active application understanding using TF-IDF + Linear SVM
• Object detection using YOLOv8
• Keyboard & mouse behavior analysis
• Audio context recognition
• XGBoost decision fusion
• Temporal smoothing
• Streamlit analytics dashboard
• CPU-friendly architecture

**System Architecture**
<img width="1025" height="578" alt="image" src="https://github.com/user-attachments/assets/38b0be97-ad18-4c17-9c0b-a03d27cc26d3" />

**Dataset And Results**

<img width="517" height="125" alt="image" src="https://github.com/user-attachments/assets/6de46317-f640-4e83-adfd-5752b2bb613b" />
<img width="517" height="125" alt="image" src="https://github.com/user-attachments/assets/ba7b4614-2627-4c5d-9fcf-8c4861ff2211" />
<img width="517" height="155" alt="image" src="https://github.com/user-attachments/assets/447e0730-58a8-407b-87b6-df5c91c12ec7" />

**Output**
<img width="1366" height="523" alt="image" src="https://github.com/user-attachments/assets/193f8a15-6c15-4f68-8ae0-ae4bacf7d9cc" />
<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/6375cf76-72ec-46a7-82c0-f2ecb40dda5b" />

Short video on this project.
https://drive.google.com/file/d/1mpOd5qOy5-QtxjhqEwl8teYaxr32a9lO/view?usp=sharing






