# --- Imports ---
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression

# --- Load your dataset ---
CSV_PATH = "link_joint_description.csv"   # make sure your file is here
df = pd.read_csv(CSV_PATH)

# Clean descriptions
df["description"] = df["description"].astype(str).str.strip().str.lower()

# Normalize joint labels
def norm_joint(x):
    if pd.isna(x): return "-"
    x = str(x).strip().lower()
    if x in {"rev", "hinge", "rotate", "rotary"}: return "revolute"
    if x in {"slide", "sliding", "telescopic", "telescoping"}: return "prismatic"
    if x in {"rigid"}: return "fixed"
    if x in {"none", "na", "n/a", "-", ""}: return "-"
    return x

for col in ["joint1", "joint2"]:
    df[col] = df[col].map(norm_joint)

df["link1"] = df["link1"].astype(float)
df["link2"] = df["link2"].fillna(0).astype(float)
df.loc[df["link2"] == 0, "joint2"] = df.loc[df["link2"] == 0, "joint2"].replace({np.nan: "-"})

# --- Train models ---
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X = vectorizer.fit_transform(df["description"])

# Regressors for link lengths
reg1 = Ridge(alpha=1.0, random_state=42).fit(X, df["link1"])
reg2 = Ridge(alpha=1.0, random_state=42).fit(X, df["link2"])

# Classifiers for joint types
CLASSES = ["revolute", "prismatic", "fixed", "-"]
y1 = df["joint1"].apply(lambda x: x if x in CLASSES else "-")
y2 = df["joint2"].apply(lambda x: x if x in CLASSES else "-")
clf1 = LogisticRegression(max_iter=1000).fit(X, y1)
clf2 = LogisticRegression(max_iter=1000).fit(X, y2)

# --- Helpers ---
def extract_lengths_units(text):
    """
    Extract up to 2 numbers with units (cm, mm, m) from text.
    Returns (link1, link2) in meters.
    """
    t = text.lower()
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*(cm|mm|m)", t)
    vals = []
    for val, unit in nums:
        v = float(val)
        if unit == "cm": v /= 100.0
        elif unit == "mm": v /= 1000.0
        vals.append(v)
    if not vals: return None
    if any(w in t for w in ["both", "each", "equal", "same"]):
        return (vals[0], vals[0])
    if len(vals) == 1: return (vals[0], 0.0)
    return (vals[0], vals[1])

def predict(text):
    """Predict link lengths and joint types from text."""
    Xq = vectorizer.transform([text.lower()])
    l1 = float(reg1.predict(Xq)[0])
    l2 = float(reg2.predict(Xq)[0])
    l1 = max(0.0, min(1.0, l1))
    l2 = max(0.0, min(1.0, l2))
    lens = extract_lengths_units(text)
    if lens is not None: l1, l2 = lens
    j1 = clf1.predict(Xq)[0]
    j2 = clf2.predict(Xq)[0]
    if l2 < 1e-6: j2 = "-"
    t = text.lower()
    if any(k in t for k in ["hinge", "elbow", "rotate", "rotating", "revolve"]):
        j1 = "revolute"
        if l2 > 0: j2 = "revolute"
    if any(k in t for k in ["slide", "sliding", "prismatic", "telescop"]):
        j1 = "prismatic"
        if l2 > 0 and ("both" in t or "each" in t or "sliding joints" in t):
            j2 = "prismatic"
    return {"link1": round(l1,4), "link2": round(l2,4), "joint1": j1, "joint2": j2}

def urdf_from_pred(pred, name="demo_robot"):
    """Generate a minimal URDF string from prediction dict."""
    l1, l2, j1, j2 = pred["link1"], pred["link2"], pred["joint1"], pred["joint2"]
    def link_block(i, L):
        Lp = max(0.01, L)
        return f"""
  <link name="link_{i}">
    <visual><geometry><box size="{Lp:.3f} 0.02 0.02"/></geometry><origin xyz="{Lp/2:.3f} 0 0" rpy="0 0 0"/></visual>
    <collision><geometry><box size="{Lp:.3f} 0.02 0.02"/></geometry><origin xyz="{Lp/2:.3f} 0 0" rpy="0 0 0"/></collision>
  </link>"""
    parts = [f'<robot name="{name}">', link_block(0, l1)]
    if l2 > 0:
        parts += [link_block(1, l2),
        f"""
  <joint name="joint_0" type="{j1}">
    <parent link="link_0"/><child link="link_1"/><origin xyz="{max(0.01,l1):.3f} 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1416" upper="3.1416" effort="10" velocity="1.0"/>
  </joint>"""]
        if j2 != "-":
            parts += ["""
  <link name="end_effector"/>""",
            f"""
  <joint name="joint_1" type="{j2}">
    <parent link="link_1"/><child link="end_effector"/><origin xyz="{max(0.01,l2):.3f} 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.1416" upper="3.1416" effort="10" velocity="1.0"/>
  </joint>"""]
    parts.append("</robot>")
    return "\n".join(parts)

# --- Example usage ---
if __name__ == "__main__":
    sample = "an elbow robot with 25cm and 10cm"
    pred = predict(sample)
    print("Predictions:", pred)
    urdf = urdf_from_pred(pred, name="my_robot")
    print("\nGenerated URDF:\n", urdf)
    with open("my_robot.urdf", "w") as f:
        f.write(urdf)
    print("\nSaved URDF to my_robot.urdf")
