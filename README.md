# text-to-urdf
*A tiny “natural language → URDF” pipeline that demonstrates the core idea behind language/multimodal → digital twin generation.*

> Given a plain-English description like “a 2-link arm, both links 10cm”, the script predicts link lengths and joint types, then emits a minimal **URDF** model you can load into robotics tooling.

---

## Why this exists
Robotics specs are often written for humans (text, diagrams). Simulators (ROS/Gazebo/RViz) need **structured robot models** (e.g., URDF). This repo shows a minimal bridge:

**Text → (TF-IDF + simple ML) → Structured specs → URDF**

---

## What this generates (URDF)
URDF (Unified Robot Description Format) is an XML schema used by ROS/Gazebo/RViz to describe a robot’s kinematic structure.

This project:
- Creates **links** as simple rectangular boxes sized by predicted **length** (meters).
- Connects links with **joints** whose **type** is predicted (`revolute`, `prismatic`, `fixed`).
- Adds minimal `<origin>`, `<axis>`, and placeholder `<limit>` tags so the file is sim/vis-friendly.
- Converts units in text (`10 cm`, `0.1 m`) to meters.

Example snippet the script produces:
```xml
<robot name="my_robot">
  <link name="link_0">
    <visual><geometry><box size="0.100 0.02 0.02"/></geometry></visual>
  </link>
  <link name="link_1">
    <visual><geometry><box size="0.100 0.02 0.02"/></geometry></visual>
  </link>
  <joint name="joint_0" type="revolute">
    <parent link="link_0"/>
    <child link="link_1"/>
    <origin xyz="0.100 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.1416" upper="3.1416" effort="10" velocity="1.0"/>
  </joint>
</robot>
