"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   STUDENT PERFORMANCE IN PORTUGAL  ·  AMSE Mag1 2025-2026                  ║
║   Authors: SEKONGO Zodigui · AKANDJONA Lucrèce · SLITI Aziza               ║
║   Run :  python app.py   →   http://127.0.0.1:8051                         ║
║   Deps:  pip install dash dash-bootstrap-components plotly pandas           ║
║          openpyxl xlsxwriter statsmodels numpy                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Stdlib ────────────────────────────────────────────────────────────────────
import io, datetime, warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys
import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
from scipy import stats as _scipy_stats


######## Data download & preprocessing ────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

_perf_path = "data/student_performance.csv"
if not os.path.exists(_perf_path):
    url_mat = "https://drive.google.com/uc?id=1FelG_t8W2BqRlPzOzgolLCqsWKCvzy8f"
    url_por = "https://drive.google.com/uc?id=1jVE4cOz_92EJXQdsedir7lxbSD81wzKj"
    mat = pd.read_csv(url_mat, sep=";")
    por = pd.read_csv(url_por, sep=";")
    mat.to_csv("data/student_mat.csv", index=False)
    por.to_csv("data/student_por.csv", index=False)
    df = pd.concat([mat, por], axis=0)
    df.to_csv(_perf_path, index=False)
    print("Ok, datasets downloaded and saved in data/")
else:
    print("Ok, datasets already present in data/")

####################################################################
# ╔══════════════════════════════════════════════════════════════════╗
# ║  1.  DATA LOADING & FEATURE ENGINEERING                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_data(path: str = "data/student_performance.csv") -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")

    # ── Human-readable labels ────────────────────────────────────────
    df["school_lbl"]    = df["school"].map({"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"})
    df["sex_lbl"]       = df["sex"].map({"F": "Female", "M": "Male"})
    df["address_lbl"]   = df["address"].map({"U": "Urban", "R": "Rural"})
    df["famsize_lbl"]   = df["famsize"].map({"LE3": "≤ 3 members", "GT3": "> 3 members"})
    df["Pstatus_lbl"]   = df["Pstatus"].map({"T": "Living together", "A": "Apart"})
    df["studytime_lbl"] = df["studytime"].map({1: "< 2 h/week", 2: "2–5 h/week",
                                                3: "5–10 h/week", 4: "> 10 h/week"})
    df["traveltime_lbl"]= df["traveltime"].map({1: "< 15 min", 2: "15–30 min",
                                                 3: "30–60 min", 4: "> 60 min"})
    df["Walc_lbl"]      = df["Walc"].map({1: "Very Low", 2: "Low",
                                           3: "Medium", 4: "High", 5: "Very High"})
    df["Dalc_lbl"]      = df["Dalc"].map({1: "Very Low", 2: "Low",
                                           3: "Medium", 4: "High", 5: "Very High"})
    df["health_lbl"]    = df["health"].map({1: "Very Bad", 2: "Bad",
                                             3: "Fair", 4: "Good", 5: "Very Good"})
    df["Medu_lbl"]      = df["Medu"].map({0: "None", 1: "Primary (4th)",
                                           2: "5th–9th", 3: "Secondary", 4: "Higher"})
    df["Fedu_lbl"]      = df["Fedu"].map({0: "None", 1: "Primary (4th)",
                                           2: "5th–9th", 3: "Secondary", 4: "Higher"})
    df["reason_lbl"]    = df["reason"].map({"home": "Proximity", "reputation": "Reputation",
                                             "course": "Course offer", "other": "Other"})
    df["guardian_lbl"]  = df["guardian"].map({"mother": "Mother", "father": "Father",
                                               "other": "Other"})

    # ── Binary Yes/No fields ─────────────────────────────────────────
    for col in ["schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]:
        df[col+"_lbl"] = df[col].map({"yes": "Yes", "no": "No"})

    # ── Derived / engineered features ────────────────────────────────
    df["grade_avg"]    = (df["G1"] + df["G2"] + df["G3"]) / 3
    df["grade_trend"]  = df["G3"] - df["G1"]          # positive = improving
    df["alc_total"]    = df["Dalc"] + df["Walc"]       # combined alcohol score
    df["parent_edu"]   = (df["Medu"] + df["Fedu"]) / 2

    df["success"] = pd.cut(
        df["G3"],
        bins=[-1, 9, 13, 20],
        labels=["At Risk  (<10)", "Average  (10–13)", "Excellent  (≥14)"]
    ).astype(str)

    df["risk_flag"] = df["G3"] < 10

    # ── Risk score (0-100) — weighted sum of risk factors ────────────
    rs = pd.Series(0, index=df.index)
    rs += (df["failures"] > 0).astype(int) * 30
    rs += (df["absences"] > 10).astype(int) * 20
    rs += (df["Walc"] >= 4).astype(int) * 20
    rs += (df["studytime"] <= 1).astype(int) * 15
    rs += (df["higher"] == "no").astype(int) * 15
    df["risk_score"] = rs.clip(0, 100)

    return df


df_full = load_data()

GLOBAL_G3_MEAN   = df_full["G3"].mean()
GLOBAL_ABS_MEAN  = df_full["absences"].mean()
GLOBAL_FAIL_RATE = (df_full["G3"] < 10).mean() * 100
AGE_MIN          = int(df_full["age"].min())
AGE_MAX          = int(df_full["age"].max())
N_TOTAL          = len(df_full)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2.  DESIGN TOKENS                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

G1 = "#2d6a4f"   # forest green  – primary
G2 = "#52b788"   # mid green
G3c = "#d8f3dc"  # pale green
O1 = "#e76f51"   # terracotta    – accent / warning
O2 = "#f4a261"   # warm orange
O3 = "#fde8d8"   # pale orange
B1 = "#6366f1"   # indigo        – info
T1 = "#14b8a6"   # teal          – neutral
BG = "#f7f9f7"
WH = "#ffffff"
BD = "#e5e7eb"
TX = "#111827"
MU = "#6b7280"
FN = "DM Sans, Helvetica Neue, sans-serif"

SEQ5  = [G1, G2, O2, O1, "#c83c1e"]         # 5-step traffic-light
CATCL = [G1, O1, B1, T1, O2, G2, "#a855f7"] # categorical palette

BASE = dict(
    font_family=FN,
    plot_bgcolor=WH, paper_bgcolor=WH,
    margin=dict(t=10, b=36, l=46, r=16),
    colorway=CATCL,
    legend=dict(orientation="h", yanchor="top", y=-0.18,
                xanchor="center", x=0.5, font_size=11,
                bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#f0f0f0", linecolor=BD,
               tickfont_size=11, title_font_size=11),
    yaxis=dict(gridcolor="#f0f0f0", linecolor=BD,
               tickfont_size=11, title_font_size=11),
)

def L(**kw):
    """Merge BASE layout with extra overrides. Strips 'title' — shown in HTML card header."""
    out = BASE.copy()
    kw.pop("title", None)   # title already rendered in .ct HTML element
    out.update(kw)
    return out

def empty_fig(msg="No data for current filters"):
    fig = go.Figure()
    fig.update_layout(**L(),
                      annotations=[dict(text=msg, xref="paper", yref="paper",
                                        x=.5, y=.5, showarrow=False,
                                        font=dict(color=MU, size=13))])
    return fig


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3.  CHART BUILDERS  (one function per chart)                   ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── 3a. Demographics ─────────────────────────────────────────────────────────

def fig_school_pie(dff):
    if dff.empty: return empty_fig()
    c = dff["school_lbl"].value_counts().reset_index()
    c.columns = ["School", "Count"]
    fig = px.pie(c, names="School", values="Count",
                 color_discrete_sequence=[G1, O1],
                 hole=0.55)
    fig.update_traces(textinfo="label+percent", textfont_size=12,
                      marker=dict(line=dict(color=WH, width=2)))
    fig.update_layout(**L(title="Students per School"))
    return fig

def fig_gender_bar(dff):
    if dff.empty: return empty_fig()
    g = dff.groupby(["sex_lbl","success"]).size().reset_index(name="n")
    fig = px.bar(g, x="sex_lbl", y="n", color="success",
                 color_discrete_map={"At Risk  (<10)":O1,
                                     "Average  (10–13)":O2,
                                     "Excellent  (≥14)":G1},
                 labels={"sex_lbl":"Gender","n":"Students","success":"Category"},
                 barmode="stack")
    fig.update_layout(**L(title="Gender × Performance"), showlegend=True)
    fig.update_traces(marker_line_width=0)
    return fig

def fig_age_hist(dff):
    if dff.empty: return empty_fig()
    fig = px.histogram(dff, x="age", nbins=(AGE_MAX-AGE_MIN+1),
                       color="sex_lbl",
                       color_discrete_map={"Female":G1,"Male":O1},
                       barmode="overlay", opacity=0.75,
                       labels={"age":"Age","count":"Students","sex_lbl":"Gender"})
    fig.update_layout(**L(title="Age Distribution by Gender"))
    fig.update_traces(marker_line_width=0)
    return fig

def fig_address_famsize(dff):
    if dff.empty: return empty_fig()
    g = dff.groupby(["address_lbl","famsize_lbl"])["G3"].mean().reset_index()
    fig = px.bar(g, x="address_lbl", y="G3", color="famsize_lbl",
                 barmode="group",
                 color_discrete_sequence=[G1, O1],
                 labels={"address_lbl":"Location","G3":"Avg G3","famsize_lbl":"Family size"})
    fig.update_layout(**L(title="Avg Grade · Location × Family Size"), yaxis_range=[0,16])
    fig.update_traces(marker_line_width=0)
    return fig

# ── 3b. Parental background ───────────────────────────────────────────────────

def fig_parent_edu(dff):
    if dff.empty: return empty_fig()
    order = ["None","Primary (4th)","5th–9th","Secondary","Higher"]
    gm = dff.groupby("Medu_lbl")["G3"].mean().reindex(order).reset_index()
    gf = dff.groupby("Fedu_lbl")["G3"].mean().reindex(order).reset_index()
    fig = go.Figure() 
    fig.add_trace(go.Bar(x=order, y=gm["G3"].round(2),
                         name="Mother", marker_color=G1, marker_line_width=0))
    fig.add_trace(go.Bar(x=order, y=gf["G3"].round(2),
                         name="Father", marker_color=O1, marker_line_width=0,
                         opacity=0.85))
    fig.update_layout(**L(title="Parent Education vs Avg G3"), barmode="group",
                      yaxis_range=[0,16],
                      xaxis_title="Education Level", yaxis_title="Avg G3")
    return fig

def fig_mjob_fjob(dff):
    if dff.empty: return empty_fig()
    jm = dff.groupby("Mjob")["G3"].mean().reset_index()
    jm["Parent"] = "Mother"
    jf = dff.groupby("Fjob")["G3"].mean().reset_index()
    jf.columns = ["Mjob","G3"]
    jf["Parent"] = "Father"
    jall = pd.concat([jm, jf], ignore_index=True)
    fig = px.bar(jall, x="Mjob", y="G3", color="Parent",
                 barmode="group",
                 color_discrete_map={"Mother":G1,"Father":O1},
                 labels={"Mjob":"Job","G3":"Avg G3"})
    fig.update_layout(**L(title="Parent Job vs Avg G3"), yaxis_range=[0,16])
    fig.update_traces(marker_line_width=0)
    return fig

def fig_guardian_reason(dff):
    if dff.empty: return empty_fig()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Guardian type", "School choice reason"])
    g = dff["guardian_lbl"].value_counts()
    fig.add_trace(go.Bar(x=g.index.tolist(), y=g.values.tolist(),
                         marker_color=[G1,O1,B1][:len(g)],
                         showlegend=False, marker_line_width=0), row=1, col=1)
    r = dff["reason_lbl"].value_counts()
    fig.add_trace(go.Bar(x=r.index.tolist(), y=r.values.tolist(),
                         marker_color=[G1,O1,B1,T1][:len(r)],
                         showlegend=False, marker_line_width=0), row=1, col=2)
    fig.update_layout(**L(title="Guardian & School Choice Reason"))
    return fig

# ── 3c. Academic support ──────────────────────────────────────────────────────

def fig_support_impact(dff):
    """Grouped box: G3 by schoolsup / famsup / paid."""
    if dff.empty: return empty_fig()
    fig = go.Figure()
    for col, lbl, color in [("schoolsup","School\nSupport",G1),
                             ("famsup","Family\nSupport",O1),
                             ("paid","Paid\nClasses",B1)]:
        for yn in ["yes","no"]:
            sub = dff[dff[col]==yn]["G3"]
            fig.add_trace(go.Box(y=sub, name=f"{lbl}={yn.capitalize()}",
                                 marker_color=color if yn=="yes" else O2,
                                 boxmean=True, line_width=1.5,
                                 showlegend=True))
    fig.update_layout(**L(title="Academic Support vs G3"),
                      yaxis_range=[0,21], yaxis_title="G3")
    return fig

def fig_studytime_violin(dff):
    if dff.empty: return empty_fig()
    order = ["< 2 h/week","2–5 h/week","5–10 h/week","> 10 h/week"]
    colors = [O1, O2, G2, G1]
    fig = go.Figure()
    for i, cat in enumerate(order):
        sub = dff[dff["studytime_lbl"]==cat]["G3"]
        if sub.empty: continue
        fig.add_trace(go.Violin(y=sub, name=cat, box_visible=True,
                                meanline_visible=True, points="outliers",
                                fillcolor=colors[i], line_color=colors[i],
                                opacity=0.78))
    fig.add_hline(y=10, line_dash="dot", line_color=MU, line_width=1,
                  annotation_text="Pass ≥10", annotation_font_size=10)
    fig.update_layout(**L(title="G3 Distribution by Weekly Study Time"),
                      showlegend=False, yaxis_range=[-1,21], yaxis_title="G3")
    return fig

def fig_travel_failures(dff):
    if dff.empty: return empty_fig()
    order = ["< 15 min","15–30 min","30–60 min","> 60 min"]
    g = dff.groupby(["traveltime_lbl","failures"])["G3"].mean().reset_index()
    g["failures_str"] = g["failures"].astype(str) + " failure(s)"
    fig = px.line(g, x="traveltime_lbl", y="G3", color="failures_str",
                  category_orders={"traveltime_lbl": order},
                  markers=True,
                  color_discrete_sequence=CATCL,
                  labels={"traveltime_lbl":"Travel Time","G3":"Avg G3"})
    fig.update_layout(**L(title="Travel Time × Past Failures → Avg G3"),
                      yaxis_range=[0,16])
    return fig

def fig_activities_internet(dff):
    if dff.empty: return empty_fig()
    labels = ["Activities","Internet","Higher\nAspiration","Romantic\nRelation"]
    cols   = ["activities","internet","higher","romantic"]
    yes_m  = [dff[dff[c]=="yes"]["G3"].mean() for c in cols]
    no_m   = [dff[dff[c]=="no"]["G3"].mean()  for c in cols]
    fig = go.Figure([
        go.Bar(name="Yes", x=labels, y=[round(v,2) for v in yes_m],
               marker_color=G1, marker_line_width=0),
        go.Bar(name="No",  x=labels, y=[round(v,2) for v in no_m],
               marker_color=O1, marker_line_width=0),
    ])
    fig.update_layout(**L(title="Lifestyle Binary Factors → Avg G3"),
                      barmode="group", yaxis_range=[0,16], yaxis_title="Avg G3")
    return fig

# ── 3d. Health & well-being ───────────────────────────────────────────────────

def fig_alcohol_heatmap(dff):
    """2-D heatmap: Dalc (workday) × Walc (weekend) → avg G3."""
    if dff.empty: return empty_fig()
    pivot = dff.pivot_table(index="Dalc", columns="Walc",
                            values="G3", aggfunc="mean")
    fig = px.imshow(pivot.round(2), text_auto=True, aspect="auto",
                    color_continuous_scale=[[0,O1],[0.5,"#fffde7"],[1,G1]],
                    zmin=6, zmax=16,
                    labels={"x":"Weekend Alcohol","y":"Workday Alcohol","color":"Avg G3"})
    fig.update_layout(**L(title="Alcohol Heatmap: Workday × Weekend → Avg G3"),
                      coloraxis_colorbar_thickness=12)
    return fig

def fig_health_freetime(dff):
    if dff.empty: return empty_fig()
    fig = px.density_heatmap(dff, x="health", y="freetime", z="G3",
                              histfunc="avg",
                              color_continuous_scale=[[0,O1],[0.5,"#fffde7"],[1,G1]],
                              labels={"health":"Health (1=Bad→5=Great)",
                                      "freetime":"Free Time (1=Low→5=High)",
                                      "G3":"Avg G3"})
    fig.update_layout(**L(title="Health × Free Time → Avg G3"),
                      coloraxis_colorbar_thickness=12)
    return fig

def fig_goout_gograde(dff):
    if dff.empty: return empty_fig()
    g = dff.groupby("goout")["G3"].agg(["mean","std","count"]).reset_index()
    fig = go.Figure([
        go.Bar(x=g["goout"], y=g["mean"].round(2),
               error_y=dict(type="data", array=g["std"].round(2), visible=True),
               marker_color=G1, marker_line_width=0, name="Avg G3"),
    ])
    fig.add_hline(y=10, line_dash="dot", line_color=O1,
                  annotation_text="Pass line", annotation_font_size=10)
    fig.update_layout(**L(title="Going-Out Frequency vs Avg G3"),
                      xaxis_title="Going Out (1=Rarely→5=Very Often)",
                      yaxis_range=[0,16], yaxis_title="Avg G3")
    return fig

# ── 3e. Grades deep-dive ─────────────────────────────────────────────────────

def fig_grade_hist(dff):
    if dff.empty: return empty_fig()
    avg = dff["G3"].mean()
    fig = px.histogram(dff, x="G3", nbins=21,
                       color_discrete_sequence=[G1],
                       labels={"G3":"Final Grade (G3)","count":"Students"})
    fig.update_traces(
        marker_color=[O1 if i < 10 else G1 for i in range(21)],
        marker_line_width=0
    )
    fig.add_vline(x=avg, line_dash="dash", line_color=O1,
                  annotation_text=f"Mean {avg:.1f}", annotation_font_size=11)
    fig.add_vline(x=10, line_dash="dot", line_color=MU,
                  annotation_text="Pass ≥10", annotation_position="top left",
                  annotation_font_size=10)
    fig.update_layout(**L(title="Final Grade Distribution"))
    return fig

def fig_grade_evolution(dff):
    if dff.empty: return empty_fig()
    per = ["G1","G2","G3"]
    means   = [dff[p].mean() for p in per]
    medians = [dff[p].median() for p in per]
    q25 = [dff[p].quantile(0.25) for p in per]
    q75 = [dff[p].quantile(0.75) for p in per]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=per+per[::-1],
        y=q75+q25[::-1],
        fill="toself", fillcolor="rgba(45,106,79,0.12)",
        line=dict(width=0), showlegend=False, name="IQR"))
    fig.add_trace(go.Scatter(x=per, y=[round(v,2) for v in means],
                             mode="lines+markers", name="Mean",
                             line=dict(color=G1,width=2.5),
                             marker=dict(size=9,color=G1,
                                         line=dict(width=2,color=WH))))
    fig.add_trace(go.Scatter(x=per, y=[round(v,2) for v in medians],
                             mode="lines+markers", name="Median",
                             line=dict(color=O1,width=2,dash="dash"),
                             marker=dict(size=7,color=O1)))
    fig.add_hline(y=10, line_dash="dot", line_color=MU, line_width=1,
                  annotation_text="Pass threshold", annotation_font_size=10)
    fig.update_layout(**L(title="Grade Evolution G1 → G2 → G3  (IQR shaded)"),
                      yaxis_range=[0,20])
    return fig

def fig_grade_trend(dff):
    if dff.empty: return empty_fig()
    dff = dff.copy()
    dff["trend_cat"] = pd.cut(dff["grade_trend"],
                               bins=[-21,-3,-0.5,0.5,3,21],
                               labels=["Strong drop","Drop","Stable","Rise","Strong rise"])
    c = dff["trend_cat"].value_counts().reset_index()
    c.columns = ["Trend","Count"]
    order = ["Strong drop","Drop","Stable","Rise","Strong rise"]
    color_map = {"Strong drop":O1,"Drop":O2,"Stable":MU,"Rise":G2,"Strong rise":G1}
    fig = px.bar(c, x="Trend", y="Count", color="Trend",
                 category_orders={"Trend":order},
                 color_discrete_map=color_map)
    fig.update_layout(**L(title="Grade Trend G1 → G3"), showlegend=False)
    fig.update_traces(marker_line_width=0)
    return fig

def fig_scatter_absences(dff):
    if dff.empty: return empty_fig()
    color_map = {"At Risk  (<10)":O1,"Average  (10–13)":O2,"Excellent  (≥14)":G1}
    fig = px.scatter(dff, x="absences", y="G3", color="success",
                     color_discrete_map=color_map,
                     trendline="ols", trendline_scope="overall",
                     trendline_color_override=MU,
                     opacity=0.65, size_max=10,
                     labels={"absences":"Absences","G3":"Final Grade (G3)","success":"Category"})
    fig.update_traces(marker_size=7, selector=dict(mode="markers"))
    fig.update_layout(**L(title="Absences vs Final Grade (OLS trend)"),
                      yaxis_range=[-1,21])
    return fig

def fig_correlation_heatmap(dff):
    if dff.empty: return empty_fig()
    cols = ["age","Medu","Fedu","traveltime","studytime","failures",
            "famrel","freetime","goout","Dalc","Walc","health",
            "absences","G1","G2","G3"]
    cols = [c for c in cols if c in dff.columns]
    corr = dff[cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale=[[0,O1],[0.5,"#f9fafb"],[1,G1]],
                    zmin=-1, zmax=1)
    fig.update_traces(textfont_size=9)
    fig.update_layout(**L(title="Correlation Matrix — Numeric Variables"),
                      height=520, coloraxis_colorbar_thickness=12)
    return fig

def fig_g3_by_variable(dff, var, lbl):
    """Generic box plot: G3 grouped by any categorical variable."""
    if dff.empty or var not in dff.columns: return empty_fig()
    order_col = lbl if lbl in dff.columns else var
    cats = dff[order_col].dropna().unique().tolist()
    fig = px.box(dff, x=order_col, y="G3", color=order_col,
                 color_discrete_sequence=CATCL,
                 labels={order_col: var, "G3":"Final Grade"},
                 notched=True)
    fig.add_hline(y=10, line_dash="dot", line_color=MU, line_width=1)
    fig.update_layout(**L(title=f"{var} → G3 Distribution"), showlegend=False)
    return fig

def fig_radar_profile(dff):
    """Radar chart comparing Excellent vs At-Risk student profiles."""
    if dff.empty: return empty_fig()
    axes = ["studytime","famrel","health","freetime","Medu","Fedu"]
    labels = ["Study Time","Family Rel.","Health","Free Time","Mother Edu","Father Edu"]
    exc = dff[dff["G3"]>=14][axes].mean().tolist()
    risk= dff[dff["G3"]<10][axes].mean().tolist()
    fig = go.Figure()
    for vals, name, color in [(exc,"Excellent (≥14)",G1),(risk,"At Risk (<10)",O1)]:
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=labels+[labels[0]],
                                       fill="toself", name=name,
                                       line_color=color,
                                       fillcolor=f"{'rgba(45,106,79,0.15)' if color==G1 else 'rgba(231,111,81,0.15)'}"))
    fig.update_layout(**L(title="Student Profile: Excellent vs At-Risk"),
                      polar=dict(radialaxis=dict(visible=True, range=[0,5])),
                      showlegend=True)
    return fig

def fig_success_donut(dff):
    if dff.empty: return empty_fig()
    c = dff["success"].value_counts().reset_index()
    c.columns = ["Category","Count"]
    color_map = {"At Risk  (<10)":O1,"Average  (10–13)":O2,"Excellent  (≥14)":G1}
    fig = px.pie(c, names="Category", values="Count",
                 color="Category", color_discrete_map=color_map,
                 hole=0.60)
    fig.update_traces(textinfo="label+percent", textfont_size=11,
                      marker=dict(line=dict(color=WH,width=2)))
    fig.update_layout(**L(title="Success Breakdown"), showlegend=False)
    return fig

def fig_correlation_bars(dff):
    """Horizontal bar chart: correlation of each variable with G3."""
    if dff.empty: return empty_fig()
    num_cols = ["age","Medu","Fedu","traveltime","studytime","failures",
                "famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2"]
    num_cols = [c for c in num_cols if c in dff.columns]
    label_map = {
        "G1":"G1 (Term 1)","G2":"G2 (Term 2)","studytime":"Study time",
        "failures":"Past failures","absences":"Absences","Walc":"Weekend alcohol",
        "Dalc":"Workday alcohol","Medu":"Mother educ.","Fedu":"Father educ.",
        "health":"Health","famrel":"Family relations","freetime":"Free time",
        "goout":"Going out","traveltime":"Travel time","age":"Age",
    }
    corrs = [(label_map.get(c,c), round(dff[c].corr(dff["G3"]),3)) for c in num_cols]
    corrs.sort(key=lambda x: x[1])
    labs, vals = zip(*corrs)
    colors = [G1 if v>=0 else O1 for v in vals]
    fig = go.Figure(go.Bar(x=list(vals), y=list(labs),
                           orientation="h",
                           marker_color=colors,
                           marker_line_width=0))
    fig.add_vline(x=0, line_color=BD, line_width=1)
    fig.update_layout(**L(title="Pearson Correlation with Final Grade (G3)"),
                      height=440, xaxis_range=[-1,1],
                      xaxis_title="Correlation r", yaxis_title="")
    return fig


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4.  HELPER UI BUILDERS                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def section_title(icon, text, color=G1):
    return html.Div([
        html.I(className=f"fas {icon}", style={"color":color,"fontSize":"13px"}),
        html.Span(f"  {text}", style={"marginLeft":"8px"}),
    ], style={
        "fontSize":"11px","fontWeight":"700","textTransform":"uppercase",
        "letterSpacing":"1.5px","color":MU,"margin":"32px 0 14px",
        "display":"flex","alignItems":"center","borderBottom":f"1px solid {BD}",
        "paddingBottom":"8px"
    })

def kpi(val_id, cmp_id, icon, color_class, label, trend_id=None):
    return dbc.Col(html.Div([
        html.Div(html.I(className=f"fas {icon}"), className=f"kpi-icon {color_class}"),
        html.Div("—", id=val_id, className="kpi-value"),
        html.Div(label, className="kpi-label"),
        html.Div("—", id=cmp_id, className="kpi-compare"),
    ], className="kpi-card"), xs=12, sm=6, lg=3)

def insight_box(content, icon="fa-lightbulb"):
    return html.Div([
        html.I(className=f"fas {icon} me-2"),
        html.Span(content),
    ], className="ibox")

def comment_box(cid):
    """Small italic comment div rendered below a chart, updated by callbacks."""
    return html.Div(id=cid, className="cmt")

def gcol(chart_id, title, subtitle="", width=6, cfg=None, height=300, comment_id=None):
    cfg = cfg or {"displayModeBar": "hover", "displaylogo": False}
    children = [
        html.Div([
            html.P(title, className="ct"),
            html.P(subtitle, className="cs"),
        ], className="ch"),
        dcc.Loading(
            dcc.Graph(id=chart_id, config=cfg,
                      style={"minHeight": f"{height}px"}),
            type="circle",
            color=G1,
            delay_show=200,
            style={"minHeight": f"{height}px"},
        ),
    ]
    if comment_id:
        children.append(comment_box(comment_id))
    return dbc.Col(html.Div(children, className="cc"), xs=12, md=width)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4b. CHART COMMENT GENERATORS                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def cmt_grade_hist(d):
    if d.empty: return ""
    mean = d["G3"].mean()
    fail_pct = (d["G3"] < 10).mean() * 100
    exc_pct  = (d["G3"] >= 14).mean() * 100
    return (f"Mean grade: {mean:.1f}/20  ·  {fail_pct:.0f}% at risk (G3<10)"
            f"  ·  {exc_pct:.0f}% excellent (G3≥14)")

def cmt_grade_evo(d):
    if d.empty: return ""
    g1m, g3m = d["G1"].mean(), d["G3"].mean()
    delta = g3m - g1m
    direction = "improved" if delta > 0.2 else "declined" if delta < -0.2 else "stable"
    return (f"Overall trend: {direction}  ·  G1 avg={g1m:.1f}  →  G3 avg={g3m:.1f}"
            f"  (Δ{delta:+.1f} pts)  ·  Median G3={d['G3'].median():.1f}")

def cmt_scatter_abs(d):
    if d.empty: return ""
    r = d["absences"].corr(d["G3"])
    strength = "moderate" if abs(r) > 0.2 else "weak"
    return (f"Pearson r(absences, G3) = {r:.3f}  —  {strength} negative relationship"
            f"  ·  Avg absences: {d['absences'].mean():.1f}")

def cmt_corr_bars(d):
    if d.empty: return ""
    label_map = {
        "G1":"G1", "G2":"G2", "studytime":"study time", "failures":"failures",
        "absences":"absences", "Walc":"weekend alcohol", "Dalc":"workday alcohol",
        "Medu":"mother edu", "Fedu":"father edu", "health":"health",
    }
    cols = [c for c in label_map if c in d.columns]
    corrs = sorted([(label_map[c], d[c].corr(d["G3"])) for c in cols],
                   key=lambda x: abs(x[1]), reverse=True)
    parts = [f"{n} (r={v:.2f})" for n, v in corrs[:3]]
    return "Top 3 predictors of G3: " + "  ·  ".join(parts)

def cmt_gender_bar(d):
    if d.empty: return ""
    gm_s = d[d["sex_lbl"]=="Male"]["G3"].dropna()
    gf_s = d[d["sex_lbl"]=="Female"]["G3"].dropna()
    gm, gf = gm_s.mean(), gf_s.mean()
    better = "Males" if gm > gf + 0.1 else "Females" if gf > gm + 0.1 else "Equal"
    if len(gm_s) >= 2 and len(gf_s) >= 2:
        pval = _scipy_stats.ttest_ind(gm_s, gf_s, equal_var=False).pvalue
        sig  = "p<0.05 ✓" if pval < 0.05 else f"p={pval:.3f} (ns)"
    else:
        sig = "n too small"
    return (f"Male avg: {gm:.1f}  ·  Female avg: {gf:.1f}"
            f"  ·  Higher: {better} (Δ{abs(gm-gf):.1f} pts)"
            f"  ·  Welch t-test: {sig}")

def cmt_addr_fam(d):
    if d.empty: return ""
    gu = d[d["address_lbl"]=="Urban"]["G3"].dropna()
    gr = d[d["address_lbl"]=="Rural"]["G3"].dropna()
    mu, mr = gu.mean(), gr.mean()
    if len(gu) >= 2 and len(gr) >= 2:
        pval = _scipy_stats.ttest_ind(gu, gr, equal_var=False).pvalue
        sig  = "p<0.05 ✓" if pval < 0.05 else f"p={pval:.3f} (ns)"
    else:
        sig = "n too small"
    return (f"Urban avg G3: {mu:.1f}  ·  Rural avg G3: {mr:.1f}"
            f"  ·  Gap: {mu-mr:+.1f} pts  ·  Welch t-test: {sig}")

def cmt_parent_edu(d):
    if d.empty: return ""
    mu_low  = d[d["Medu_lbl"].isin(["None","Primary (4th)"])]["G3"].mean()
    mu_high = d[d["Medu_lbl"] == "Higher"]["G3"].mean()
    r = d["Medu"].corr(d["G3"])
    if np.isnan(mu_low) or np.isnan(mu_high): return ""
    return (f"Mother higher edu → avg G3={mu_high:.1f}  vs  primary={mu_low:.1f}"
            f"  (Δ{mu_high-mu_low:+.1f} pts)  ·  r(Medu, G3)={r:.2f}")

def cmt_studytime(d):
    if d.empty: return ""
    hi_g = d[d["studytime"] >= 3]["G3"].dropna()
    lo_g = d[d["studytime"] <= 1]["G3"].dropna()
    hi, lo = hi_g.mean(), lo_g.mean()
    if np.isnan(hi) or np.isnan(lo): return ""
    if len(hi_g) >= 2 and len(lo_g) >= 2:
        pval = _scipy_stats.ttest_ind(hi_g, lo_g, equal_var=False).pvalue
        sig  = "p<0.05 ✓" if pval < 0.05 else f"p={pval:.3f} (ns)"
    else:
        sig = "n too small"
    return (f">5 h/week: avg G3={hi:.1f}  ·  <2 h/week: avg G3={lo:.1f}"
            f"  ·  Bonus: {hi-lo:+.1f} pts  ·  Welch t-test: {sig}")

def cmt_failures(d):
    if d.empty: return ""
    g0 = d[d["failures"] == 0]["G3"].dropna()
    g1p = d[d["failures"]  > 0]["G3"].dropna()
    f0, f1 = g0.mean(), g1p.mean()
    if np.isnan(f0) or np.isnan(f1): return ""
    n_risk = (d["failures"] > 0).sum()
    if len(g0) >= 2 and len(g1p) >= 2:
        pval = _scipy_stats.ttest_ind(g0, g1p, equal_var=False).pvalue
        sig  = "p<0.05 ✓" if pval < 0.05 else f"p={pval:.3f} (ns)"
    else:
        sig = "n too small"
    return (f"0 failures: avg G3={f0:.1f}  ·  ≥1 failure: avg G3={f1:.1f}"
            f"  (Δ{f0-f1:+.1f} pts)  ·  {n_risk} at risk"
            f"  ·  Welch t-test: {sig}")

def cmt_alc_heat(d):
    if d.empty: return ""
    low  = d[(d["Walc"] <= 2) & (d["Dalc"] <= 2)]["G3"].mean()
    high = d[(d["Walc"] >= 4) | (d["Dalc"] >= 4)]["G3"].mean()
    if np.isnan(low) or np.isnan(high): return ""
    return (f"Low consumption (Walc+Dalc ≤2): avg G3={low:.1f}"
            f"  ·  High consumption (≥4 either): avg G3={high:.1f}"
            f"  ·  Impact: {low-high:+.1f} pts")

def cmt_school_pie(d):
    if d.empty: return ""
    counts = d["school_lbl"].value_counts()
    parts = [f"{s}: {n} students ({n/len(d)*100:.0f}%)" for s, n in counts.items()]
    return "  ·  ".join(parts)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5.  APP & LAYOUT                                               ║
# ╚══════════════════════════════════════════════════════════════════╝

EXTERNAL = [
    dbc.themes.FLATLY,
    "https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Playfair+Display:wght@700&display=swap",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
]

app = Dash(__name__, external_stylesheets=EXTERNAL,
           suppress_callback_exceptions=True)
app.title = "Student Performance · Portugal"
server = app.server   # Gunicorn entry point

# ── Ticker text ───────────────────────────────────────────────────────────────
_tick = (
    f"📊 {N_TOTAL} students · 2 schools · 33 variables  "
    f"|  Global avg grade: {GLOBAL_G3_MEAN:.1f}/20  "
    f"| ⚠️ At-risk (G3<10): {(df_full['G3']<10).mean()*100:.0f}%  "
    f"| 📚 Study time >5h/week → +3 points avg  "
    f"|  High alcohol → −4 points avg  "
    f"|  Mother's education is the #1 family predictor  "
)
TICK = _tick * 2

CSS = f"""
/* ── Fonts & Base ───────────────────────── */
*{{box-sizing:border-box;}}
body{{font-family:'DM Sans',Helvetica,sans-serif;background:{BG};color:{TX};margin:0;padding:0;}}

/* ── Ticker ─────────────────────────────── */
.ticker-wrap{{background:{G1};padding:9px 24px;display:flex;align-items:center;gap:14px;overflow:hidden;}}
.ticker-lbl{{background:{O1};color:#fff;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:.6px;flex-shrink:0;}}
.ticker-txt{{white-space:nowrap;color:rgba(255,255,255,.88);font-size:12px;animation:scroll 40s linear infinite;display:inline-block;}}
@keyframes scroll{{from{{transform:translateX(0)}}to{{transform:translateX(-50%)}}}}

/* ── Top header ─────────────────────────── */
.topbar{{background:{WH};border-bottom:1px solid {BD};padding:14px 28px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:99;}}
.topbar h1{{font-size:19px;font-weight:600;color:{TX};margin:0;}}
.topbar p{{font-size:12px;color:{MU};margin:2px 0 0;}}

/* ── Filter bar ─────────────────────────── */
.fbar{{background:{WH};border-bottom:1px solid {BD};padding:12px 28px;}}

/* ── Sidebar tabs ───────────────────────── */
.page-tabs{{background:{WH};border-bottom:2px solid {BD};padding:0 28px;display:flex;gap:0;position:sticky;top:64px;z-index:98;}}
.ptab{{padding:12px 18px;font-size:13px;font-weight:500;color:{MU};cursor:pointer;border:none;background:none;border-bottom:3px solid transparent;margin-bottom:-2px;transition:all .15s;font-family:'DM Sans',sans-serif;}}
.ptab:hover{{color:{TX};}}
.ptab.active{{color:{G1};border-bottom-color:{G1};font-weight:600;}}

/* ── Content wrapper ────────────────────── */
.content{{padding:0 28px 40px;}}

/* ── KPI cards ──────────────────────────── */
.kpi-card{{background:{WH};border:1px solid {BD};border-radius:14px;padding:18px 20px;height:100%;position:relative;overflow:hidden;transition:box-shadow .2s;}}
.kpi-card:hover{{box-shadow:0 4px 20px rgba(0,0,0,.06);}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.kpi-card.green::before{{background:{G2};}}
.kpi-card.blue::before{{background:{B1};}}
.kpi-card.orange::before{{background:{O1};}}
.kpi-card.teal::before{{background:{T1};}}
.kpi-card.purple::before{{background:#a855f7;}}
.kpi-icon{{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:14px;margin-bottom:10px;}}
.kpi-icon.green{{background:{G3c};color:{G1};}}
.kpi-icon.blue{{background:#ede9fe;color:#7c3aed;}}
.kpi-icon.orange{{background:{O3};color:{O1};}}
.kpi-icon.teal{{background:#ccfbf1;color:#0f766e;}}
.kpi-icon.purple{{background:#faf5ff;color:#9333ea;}}
.kpi-value{{font-size:28px;font-weight:700;line-height:1;margin-bottom:4px;color:{TX};}}
.kpi-label{{font-size:10px;color:{MU};font-weight:700;text-transform:uppercase;letter-spacing:.8px;}}
.kpi-compare{{font-size:11px;margin-top:8px;color:{MU};}}
.kpi-compare.up{{color:#16a34a;font-weight:500;}}
.kpi-compare.down{{color:{O1};font-weight:500;}}

/* ── Chart cards ────────────────────────── */
.cc{{background:{WH};border:1px solid {BD};border-radius:14px;overflow:hidden;margin-bottom:20px;transition:box-shadow .2s;}}
.cc:hover{{box-shadow:0 4px 20px rgba(0,0,0,.06);}}
.ch{{padding:16px 20px 0;}}
.ct{{font-size:14px;font-weight:600;color:{TX};margin:0;}}
.cs{{font-size:12px;color:{MU};margin:2px 0 0;}}
.cmt{{font-size:11.5px;color:{MU};padding:4px 20px 14px;font-style:italic;line-height:1.5;border-top:1px solid {BD};margin-top:-2px;}}

/* ── Insight box ────────────────────────── */
.ibox{{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1px solid #bbf7d0;border-radius:10px;padding:12px 16px;margin:0 20px 16px;font-size:13px;color:#15803d;}}
.ibox.warn{{background:linear-gradient(135deg,#fff7ed,#ffedd5);border-color:#fed7aa;color:#9a3412;}}

/* ── Tabs (chart-level) ─────────────────── */
.custom-tabs .tab{{border-radius:0!important;border:none!important;border-bottom:2px solid transparent!important;padding:8px 14px!important;font-size:12px!important;font-weight:500!important;color:{MU}!important;background:transparent!important;}}
.custom-tabs .tab--selected{{color:{G1}!important;border-bottom-color:{G1}!important;font-weight:600!important;}}
.custom-tabs .tab-container--top{{border-bottom:1px solid {BD}!important;margin-bottom:0!important;}}

/* ── Data table ─────────────────────────── */
.dash-table-container .dash-spreadsheet-inner td{{font-size:12px!important;}}
.dash-table-container .dash-spreadsheet-inner th{{font-size:11px!important;}}

/* ── Buttons ────────────────────────────── */
.btn-edu{{border-radius:8px!important;font-size:13px!important;font-weight:500!important;}}

/* ── Footer ─────────────────────────────── */
.footer{{background:{WH};border-top:1px solid {BD};padding:14px 28px;font-size:12px;color:{MU};display:flex;justify-content:space-between;align-items:center;}}

/* ── Finding cards ──────────────────────── */
.finding-card{{background:{WH};border:1px solid {BD};border-radius:14px;padding:20px 22px;height:100%;}}
.finding-num{{font-size:36px;font-weight:700;color:{O1};line-height:1;margin-bottom:4px;}}
.finding-lbl{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:{G1};margin-bottom:8px;}}
.finding-desc{{font-size:13px;color:{MU};line-height:1.5;}}

/* ── Risk score badges ──────────────────── */
.risk-high{{background:{O1};color:#fff;border-radius:20px;padding:3px 10px;font-size:11px;font-weight:700;display:inline-block;}}
.risk-med{{background:{O2};color:#fff;border-radius:20px;padding:3px 10px;font-size:11px;font-weight:700;display:inline-block;}}
.risk-low{{background:{G2};color:#fff;border-radius:20px;padding:3px 10px;font-size:11px;font-weight:700;display:inline-block;}}

/* ── Student profile card ───────────────── */
.profile-kv{{border-collapse:collapse;width:100%;}}
.profile-kv td{{padding:4px 14px 4px 0;font-size:12px;vertical-align:top;}}
.profile-kv td:first-child{{color:{MU};font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap;}}
"""

# Write dynamic CSS to assets/ (Dash 4 removed html.Style)
import os as _os
_os.makedirs(_os.path.join(_os.path.dirname(__file__) or ".", "assets"), exist_ok=True)
with open(_os.path.join(_os.path.dirname(__file__) or ".", "assets", "custom.css"), "w", encoding="utf-8") as _f:
    _f.write(CSS)

# ── Filter component ──────────────────────────────────────────────────────────
FILTER_BAR = html.Div(
    dbc.Row([
        dbc.Col([
            dbc.Label("🏫 School", size="sm",
                      style={"fontWeight":"600","fontSize":"12px"}),
            dcc.Dropdown(
                id="f-school",
                options=[{"label":"All Schools","value":"all"}] +
                        [{"label":s,"value":s}
                         for s in sorted(df_full["school_lbl"].dropna().unique())],
                value="all", clearable=False,
                style={"fontSize":"13px"},
            ),
        ], xs=12, sm=4, md=2),
        dbc.Col([
            dbc.Label("👤 Gender", size="sm",
                      style={"fontWeight":"600","fontSize":"12px"}),
            dcc.Dropdown(
                id="f-sex",
                options=[{"label":"All Genders","value":"all"},
                         {"label":"Female","value":"Female"},
                         {"label":"Male","value":"Male"}],
                value="all", clearable=False,
                style={"fontSize":"13px"},
            ),
        ], xs=12, sm=4, md=2),
        dbc.Col([
            dbc.Label("🏙️ Address", size="sm",
                      style={"fontWeight":"600","fontSize":"12px"}),
            dcc.Dropdown(
                id="f-address",
                options=[{"label":"All","value":"all"},
                         {"label":"Urban","value":"Urban"},
                         {"label":"Rural","value":"Rural"}],
                value="all", clearable=False,
                style={"fontSize":"13px"},
            ),
        ], xs=12, sm=4, md=2),
        dbc.Col([
            dbc.Label(
                ["📅 Age  ",
                 html.Span(id="age-disp",
                           style={"color":G1,"fontWeight":"700"})],
                size="sm", style={"fontWeight":"600","fontSize":"12px"}),
            dcc.RangeSlider(
                id="f-age", min=AGE_MIN, max=AGE_MAX, step=1,
                value=[AGE_MIN, AGE_MAX],
                marks={v:str(v) for v in range(AGE_MIN, AGE_MAX+1)},
                tooltip={"placement":"bottom","always_visible":False},
            ),
        ], xs=12, md=4),
        dbc.Col([
            dbc.Label(".", size="sm",
                      style={"color":BG,"fontSize":"12px"}),
            dbc.Button(
                [html.I(className="fas fa-undo me-2"), "Reset"],
                id="btn-reset", color="outline-success",
                size="sm", className="btn-edu d-block w-100",
            ),
        ], xs=12, sm=4, md=2),
    ], className="g-2 align-items-end"),
    className="fbar",
)

# ── Page navigation ───────────────────────────────────────────────────────────
NAV_PAGES = [
    ("overview", "fa-chart-pie", "Overview"),
    ("demographics", "fa-users", "Demographics"),
    ("parental", "fa-user-tie", "Parental"),
    ("academic", "fa-graduation-cap", "Academic"),
    ("wellbeing", "fa-heart", "Well-Being"),
    ("grades", "fa-star", "Grades"),
    ("data", "fa-table", "Data"),
]

PAGE_NAV = html.Div([
    html.Button(
        [html.I(className=f"fas {ico} me-2"), lbl],
        id=f"nav-{pid}", className="ptab active" if pid=="overview" else "ptab",
        **{"data-page": pid},
    )
    for pid, ico, lbl in NAV_PAGES
], className="page-tabs", id="page-tabs")

# ── Hidden page store ─────────────────────────────────────────────────────────
PAGE_STORE = dcc.Store(id="page-store", data="overview")


def page_overview():
    return html.Div([
        # Overview charts
        section_title("fa-chart-bar","Overview Charts"),
        dbc.Row([
            gcol("ov-grade-hist","Final Grade Distribution","Orange = fail, green = pass",
                 comment_id="ov-grade-hist-cmt"),
            gcol("ov-grade-evo","Grade Evolution G1→G2→G3","Mean, median & IQR ribbon",
                 comment_id="ov-grade-evo-cmt"),
        ], className="g-3"),
        dbc.Row([
            gcol("ov-radar","Student Profile: Excellent vs At-Risk","Spider chart comparison",6,height=340),
            gcol("ov-donut","Success Breakdown","By performance category",3,height=340),
            gcol("ov-scatter","Absences vs Final Grade","OLS trendline",3,height=340,
                 comment_id="ov-scatter-cmt"),
        ], className="g-3"),

        # Correlation
        section_title("fa-project-diagram","Correlations with G3"),
        dbc.Row([
            gcol("ov-corr-bars","Variable Impact (Pearson r)","Green = positive, orange = negative",
                 6, height=440, comment_id="ov-corr-bars-cmt"),
            gcol("ov-corr-heat","Full Correlation Matrix","All numeric variables",6,height=520),
        ], className="g-3"),
    ])


def page_demographics():
    return html.Div([
        section_title("fa-users","Student Demographics"),
        dbc.Row([
            gcol("dm-school-pie","Students per School","Distribution across both schools",3,height=280,
                 comment_id="dm-school-pie-cmt"),
            gcol("dm-gender-bar","Gender × Performance","Stacked by success category",3,height=280,
                 comment_id="dm-gender-bar-cmt"),
            gcol("dm-age-hist","Age Distribution","By gender",3,height=280),
            gcol("dm-addr-fam","Location × Family Size → Avg G3","Urban/Rural × small/large family",3,height=280,
                 comment_id="dm-addr-fam-cmt"),
        ], className="g-3"),
        section_title("fa-map-marker-alt","Geographic & Lifestyle Context"),
        dbc.Row([
            gcol("dm-trend","Grade Trend G1→G3","Students improving vs declining",4,height=300),
            gcol("dm-activities","Lifestyle Factors → Avg G3",
                 "Activities, internet, higher aspiration, romance",4,height=300),
            gcol("dm-guardian","Guardian & School Choice Reason","Side by side",4,height=300),
        ], className="g-3"),
    ])


def page_parental():
    return html.Div([
        section_title("fa-user-tie","Parental Background & Influence"),
        dbc.Row([
            gcol("pa-edu","Parent Education Level → Avg G3","Mother vs Father",6,height=320,
                 comment_id="pa-edu-cmt"),
            gcol("pa-job","Parent Job → Avg G3","Mother vs Father",6,height=320),
        ], className="g-3"),
        html.Div([
            insight_box("Mother's education is the strongest parental predictor of G3 (r≈0.22). "
                        "Students whose mothers hold higher education score ≈1.5 pts above peers."),
        ]),
        dbc.Row([
            gcol("pa-pstatus","Parent Cohabitation → G3 Box","Living together vs apart",4,height=300),
            gcol("pa-guardian2","Guardian Type → Avg G3","Mother / Father / Other",4,height=300),
            gcol("pa-famsup","Family Support → G3","Family educational support",4,height=300),
        ], className="g-3"),
    ])


def page_academic():
    return html.Div([
        section_title("fa-graduation-cap","Academic Performance & Support"),
        dbc.Row([
            gcol("ac-studytime","G3 by Weekly Study Time","Violin + box + outliers",6,height=340,
                 comment_id="ac-studytime-cmt"),
            gcol("ac-support","Support Programs → G3",
                 "School support / Family support / Paid classes",6,height=340),
        ], className="g-3"),
        dbc.Row([
            gcol("ac-travel","Travel Time × Past Failures → Avg G3","Line chart",6,height=300),
            gcol("ac-failures","Past Failures Impact","Box per failure count",6,height=300,
                 comment_id="ac-failures-cmt"),
        ], className="g-3"),
        section_title("fa-clock","Time & Lifestyle"),
        dbc.Row([
            gcol("ac-freetime","Free Time → G3","After-school free time distribution",4,height=300),
            gcol("ac-goout","Going Out → Avg G3","Social frequency effect",4,height=300),
            gcol("ac-internet","Internet Access → G3","Box comparison",4,height=300),
        ], className="g-3"),
    ])


def page_wellbeing():
    return html.Div([
        section_title("fa-heart","Health & Well-Being"),
        dbc.Row([
            gcol("wb-alc-heat","Alcohol Heatmap","Workday × Weekend → Avg G3",6,height=380,
                 comment_id="wb-alc-heat-cmt"),
            gcol("wb-health","Health × Free Time → Avg G3","Density heatmap",6,height=380),
        ], className="g-3"),
        html.Div([
            insight_box(
                "High weekend drinking (Walc 4–5) is the strongest behavioural predictor "
                "of poor performance — students score ≈4 points below low-consumption peers.",
                icon="fa-wine-bottle"),
        ]),
        dbc.Row([
            gcol("wb-goout2","Going-Out Frequency → Avg G3","With error bars",4,height=300),
            gcol("wb-romantic","Romantic Relationship → G3","Box comparison",4,height=300),
            gcol("wb-health2","Self-Rated Health → G3","Box per health rating",4,height=300),
        ], className="g-3"),
    ])


def page_grades():
    return html.Div([
        section_title("fa-star","Grade Deep-Dive"),
        dbc.Row([
            gcol("gr-hist","Final Grade (G3) Distribution","Coloured by pass/fail",6,height=300,
                 comment_id="gr-hist-cmt"),
            gcol("gr-evo","Grade Evolution G1→G2→G3","Mean, median, IQR",6,height=300,
                 comment_id="gr-evo-cmt"),
        ], className="g-3"),
        dbc.Row([
            gcol("gr-trend","Grade Trend G1→G3","Students improving / declining / stable",4,height=280),
            gcol("gr-scatter","Absences vs G3","OLS trendline",4,height=280,
                 comment_id="gr-scatter-cmt"),
            gcol("gr-donut","Success Breakdown","Excellent / Average / At-Risk",4,height=280),
        ], className="g-3"),
        section_title("fa-project-diagram","Full Correlation Analysis"),
        dbc.Row([
            gcol("gr-corr-bars","Pearson r with G3","All numeric variables",6,height=460,
                 comment_id="gr-corr-bars-cmt"),
            gcol("gr-corr-heat","Full Correlation Matrix","",6,height=540),
        ], className="g-3"),
    ])


def page_data():
    return html.Div([
        section_title("fa-table","Student Data"),
        html.Div([
            html.Div([
                html.P(id="tbl-meta", className="ct"),
                html.P("Sortable & filterable — cliquez une ligne pour voir le profil complet",
                       className="cs"),
            ], className="ch", style={"display":"flex","justifyContent":"space-between","alignItems":"start"}),
            html.Div(id="tbl-stats",
                     style={"padding":"8px 20px","fontSize":"12px","color":MU}),
            html.Div([
                dash_table.DataTable(
                    id="data-table",
                    columns=[{"name":c,"id":c} for c in
                             ["school_lbl","sex_lbl","age","address_lbl",
                              "studytime_lbl","failures","absences",
                              "G1","G2","G3","success","risk_score"]],
                    page_size=15,
                    sort_action="native",
                    filter_action="native",
                    cell_selectable=True,
                    style_table={"overflowX":"auto"},
                    style_cell={"textAlign":"left","padding":"9px 12px",
                                "fontFamily":FN,"fontSize":"12px",
                                "border":f"1px solid {BD}"},
                    style_header={
                        "backgroundColor":G1,"color":WH,
                        "fontWeight":"600","fontSize":"11px",
                        "textTransform":"uppercase","letterSpacing":"0.5px"},
                    style_data_conditional=[
                        {"if":{"row_index":"odd"},"backgroundColor":BG},
                        {"if":{"filter_query":"{G3} < 10","column_id":"G3"},
                         "color":O1,"fontWeight":"700"},
                        {"if":{"filter_query":"{G3} >= 14","column_id":"G3"},
                         "color":G1,"fontWeight":"700"},
                        {"if":{"filter_query":"{risk_score} >= 50","column_id":"risk_score"},
                         "color":O1,"fontWeight":"700"},
                    ],
                ),
            ], style={"padding":"12px 20px 20px"}),
        ], className="cc"),

        # Profile card — populated when user clicks a row
        html.Div(id="profile-card", style={"marginTop":"16px"}),

        # At-Risk section
        section_title("fa-exclamation-triangle","Students At Risk (G3 < 10)", O1),
        html.Div(id="atrisk-section"),
    ])


PAGE_MAP = {
    "overview":     page_overview,
    "demographics": page_demographics,
    "parental":     page_parental,
    "academic":     page_academic,
    "wellbeing":    page_wellbeing,
    "grades":       page_grades,
    "data":         page_data,
}

# ── Main layout ───────────────────────────────────────────────────────────────
app.layout = dbc.Container([
    PAGE_STORE,

    # Downloads
    dcc.Download(id="dl-csv"),
    dcc.Download(id="dl-excel"),
    dcc.Download(id="dl-html"),

    # Ticker
    html.Div([
        html.Span("LIVE", className="ticker-lbl"),
        html.Div(html.Span(TICK, className="ticker-txt"),
                 style={"overflow":"hidden","flex":"1"}),
    ], className="ticker-wrap"),

    # Top bar
    html.Div([
        html.Div([
            html.H1("📚 Student Performance in Portugal"),
            html.P("UCI dataset · Gabriel Pereira & Mousinho da Silveira · AMSE Mag1 2025-2026"),
        ]),
        html.Div([
            html.Span(id="filter-badge",
                      className="badge bg-warning text-dark me-2",
                      style={"fontSize":"12px"}),
            dbc.DropdownMenu([
                dbc.DropdownMenuItem([html.I(className="fas fa-file-csv me-2"),   "CSV"],   id="btn-csv"),
                dbc.DropdownMenuItem([html.I(className="fas fa-file-excel me-2"), "Excel"], id="btn-excel"),
                dbc.DropdownMenuItem([html.I(className="fas fa-code me-2"),       "HTML Report"], id="btn-html"),
            ], label=html.Span([html.I(className="fas fa-download me-2"),"Export"]),
               color="outline-secondary", size="sm", className="btn-edu me-2"),
        ], style={"display":"flex","alignItems":"center"}),
    ], className="topbar"),

    # Page nav
    PAGE_NAV,

    # Filters
    FILTER_BAR,

    # KPI + Key Findings — always in DOM, shown only on overview page
    html.Div(id="kpi-section", className="content", style={"paddingTop":"24px","display":"block"}, children=[
        section_title("fa-tachometer-alt", "Key Metrics"),
        dbc.Row([
            kpi("kpi-total","kpi-total-cmp","fa-users","green","Total Students"),
            kpi("kpi-avg",  "kpi-avg-cmp",  "fa-star", "blue", "Avg Final Grade /20"),
            kpi("kpi-fail", "kpi-fail-cmp", "fa-exclamation-triangle","orange","Failure Rate"),
            kpi("kpi-abs",  "kpi-abs-cmp",  "fa-calendar-times","teal","Avg Absences"),
        ], className="g-3 mb-2"),
        section_title("fa-lightbulb","Key Findings", O1),
        dbc.Row([
            dbc.Col(html.Div([
                html.Div("Finding 1 — Study", className="finding-lbl"),
                html.Div(id="ins1", className="finding-num"),
                html.Div("points more for >5h/week vs <2h study time", className="finding-desc"),
            ], className="finding-card"), xs=12, md=3),
            dbc.Col(html.Div([
                html.Div("Finding 2 — Alcohol", className="finding-lbl"),
                html.Div(id="ins2", className="finding-num"),
                html.Div("grade impact from high weekend alcohol (Walc 4–5)", className="finding-desc"),
            ], className="finding-card"), xs=12, md=3),
            dbc.Col(html.Div([
                html.Div("Finding 3 — Failures", className="finding-lbl"),
                html.Div(id="ins3", className="finding-num"),
                html.Div("points gap: zero vs 1+ past course failures", className="finding-desc"),
            ], className="finding-card"), xs=12, md=3),
            dbc.Col(html.Div([
                html.Div("Finding 4 — Higher Edu", className="finding-lbl"),
                html.Div(id="ins4", className="finding-num"),
                html.Div("points higher avg for students aspiring to higher education", className="finding-desc"),
            ], className="finding-card"), xs=12, md=3),
        ], className="g-3 mb-2"),
    ]),

    # Dynamic page content
    html.Div(id="page-content", className="content",
             style={"paddingTop":"8px"}),

    # Footer
    html.Footer([
        html.Span("© 2025 Student Performance Dashboard · AMSE Mag1 · Zodigui Sekongo"),
        html.Span(id="footer-r"),
    ], className="footer"),

], fluid=True, style={"padding":"0"})


# ╔══════════════════════════════════════════════════════════════════╗
# ║  6.  FILTER HELPER                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def apply_filters(school, sex, address, age_range):
    dff = df_full.copy()
    if school  != "all": dff = dff[dff["school_lbl"]  == school]
    if sex     != "all": dff = dff[dff["sex_lbl"]     == sex]
    if address != "all": dff = dff[dff["address_lbl"] == address]
    lo, hi = min(age_range), max(age_range)
    dff = dff[(dff["age"] >= lo) & (dff["age"] <= hi)]
    return dff


# ╔══════════════════════════════════════════════════════════════════╗
# ║  7.  PAGE ROUTING CALLBACK                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("page-store","data"),
    [Input(f"nav-{pid}","n_clicks") for pid,_,__ in NAV_PAGES],
    prevent_initial_call=True
)
def switch_page(*args):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return btn_id.replace("nav-","")

@app.callback(
    Output("page-content","children"),
    Output("kpi-section","style"),
    [Output(f"nav-{pid}","className") for pid,_,__ in NAV_PAGES],
    Input("page-store","data")
)
def render_page(page):
    content = PAGE_MAP.get(page, page_overview)()
    kpi_style = {"paddingTop":"24px","display":"block"} if page == "overview" else {"display":"none"}
    classes = ["ptab active" if pid==page else "ptab"
               for pid,_,__ in NAV_PAGES]
    return [content, kpi_style] + classes


# ╔══════════════════════════════════════════════════════════════════╗
# ║  8.  RESET FILTER CALLBACK                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("f-school","value"),
    Output("f-sex","value"),
    Output("f-address","value"),
    Output("f-age","value"),
    Input("btn-reset","n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return "all","all","all",[AGE_MIN,AGE_MAX]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  9.  AGE DISPLAY CALLBACK                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(Output("age-disp","children"), Input("f-age","value"))
def update_age(v): return f"{min(v)}–{max(v)} yrs"


# ╔══════════════════════════════════════════════════════════════════╗
# ║  10. GLOBAL STATS CALLBACK (KPIs + badge + footer)             ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("kpi-total","children"), Output("kpi-avg","children"),
    Output("kpi-fail","children"),  Output("kpi-abs","children"),
    Output("kpi-total-cmp","children"), Output("kpi-avg-cmp","children"),
    Output("kpi-avg-cmp","className"),  Output("kpi-fail-cmp","children"),
    Output("kpi-fail-cmp","className"), Output("kpi-abs-cmp","children"),
    Output("ins1","children"), Output("ins2","children"),
    Output("ins3","children"), Output("ins4","children"),
    Output("filter-badge","children"),
    Output("footer-r","children"),
    Input("f-school","value"), Input("f-sex","value"),
    Input("f-address","value"), Input("f-age","value"),
)
def update_stats(school, sex, address, age):
    dff = apply_filters(school, sex, address, age)
    n   = len(dff)

    g3m  = dff["G3"].mean()   if n else 0
    absm = dff["absences"].mean() if n else 0
    fp   = (dff["G3"]<10).mean()*100 if n else 0

    g3d  = (g3m - GLOBAL_G3_MEAN)   / GLOBAL_G3_MEAN   * 100 if n else 0
    fpd  = (fp  - GLOBAL_FAIL_RATE) / GLOBAL_FAIL_RATE * 100 if (n and GLOBAL_FAIL_RATE) else 0
    absd = (absm - GLOBAL_ABS_MEAN) / GLOBAL_ABS_MEAN  * 100 if n else 0

    hs = dff[dff["studytime"]>2]["G3"].mean() if n else 0
    ls = dff[dff["studytime"]<=2]["G3"].mean() if n else 0
    i1 = f"{hs-ls:+.1f}" if n and not np.isnan(hs-ls) else "N/A"

    ha = dff[dff["Walc"]>=4]["G3"].mean() if n else 0
    la = dff[dff["Walc"]<=2]["G3"].mean() if n else 0
    i2 = f"{ha-la:+.1f}" if n and not np.isnan(ha-la) else "N/A"

    hf = dff[dff["failures"]==0]["G3"].mean() if n else 0
    lf = dff[dff["failures"]>0]["G3"].mean()  if n else 0
    i3 = f"{hf-lf:+.1f}" if n and not np.isnan(hf-lf) else "N/A"

    hh = dff[dff["higher"]=="yes"]["G3"].mean() if n else 0
    lh = dff[dff["higher"]=="no"]["G3"].mean()  if n else 0
    i4 = f"{hh-lh:+.1f}" if n and not np.isnan(hh-lh) else "N/A"

    return (
        str(n),
        f"{g3m:.1f}" if n else "N/A",
        f"{fp:.1f}%" if n else "N/A",
        f"{absm:.1f}" if n else "N/A",
        f"{n/N_TOTAL*100:.0f}% of dataset",
        f"{'▲' if g3d>=0 else '▼'} vs global: {g3d:+.1f}%",
        "kpi-compare up" if g3d>=0 else "kpi-compare down",
        f"{'▲' if fpd<=0 else '▼'} vs global: {fpd:+.1f}%",
        "kpi-compare down" if fpd>0 else "kpi-compare up",
        f"{'▲' if absd<=0 else '▼'} vs global: {absd:+.1f}%",
        i1, i2, i3, i4,
        f"{n} students" if n<N_TOTAL else "All students",
        f"{n} students · 33 variables",
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  11. OVERVIEW PAGE CHARTS                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("ov-grade-hist","figure"),      Output("ov-grade-evo","figure"),
    Output("ov-radar","figure"),           Output("ov-donut","figure"),
    Output("ov-scatter","figure"),         Output("ov-corr-bars","figure"),
    Output("ov-corr-heat","figure"),
    Output("ov-grade-hist-cmt","children"), Output("ov-grade-evo-cmt","children"),
    Output("ov-scatter-cmt","children"),    Output("ov-corr-bars-cmt","children"),
    Input("f-school","value"), Input("f-sex","value"),
    Input("f-address","value"), Input("f-age","value"),
    Input("page-store","data"),
)
def upd_overview(school,sex,address,age,page):
    if page!="overview": return [dash.no_update]*11
    d = apply_filters(school,sex,address,age)
    return (fig_grade_hist(d), fig_grade_evolution(d),
            fig_radar_profile(d), fig_success_donut(d),
            fig_scatter_absences(d), fig_correlation_bars(d),
            fig_correlation_heatmap(d),
            cmt_grade_hist(d), cmt_grade_evo(d),
            cmt_scatter_abs(d), cmt_corr_bars(d))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  12. DEMOGRAPHICS PAGE CHARTS                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("dm-school-pie","figure"),       Output("dm-gender-bar","figure"),
    Output("dm-age-hist","figure"),         Output("dm-addr-fam","figure"),
    Output("dm-trend","figure"),            Output("dm-activities","figure"),
    Output("dm-guardian","figure"),
    Output("dm-school-pie-cmt","children"), Output("dm-gender-bar-cmt","children"),
    Output("dm-addr-fam-cmt","children"),
    Input("f-school","value"), Input("f-sex","value"),
    Input("f-address","value"), Input("f-age","value"),
    Input("page-store","data"),
)
def upd_demo(school,sex,address,age,page):
    if page!="demographics": return [dash.no_update]*10
    d = apply_filters(school,sex,address,age)
    return (fig_school_pie(d), fig_gender_bar(d), fig_age_hist(d),
            fig_address_famsize(d), fig_grade_trend(d),
            fig_activities_internet(d), fig_guardian_reason(d),
            cmt_school_pie(d), cmt_gender_bar(d), cmt_addr_fam(d))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  13. PARENTAL PAGE CHARTS                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("pa-edu","figure"),      Output("pa-job","figure"),
    Output("pa-pstatus","figure"),  Output("pa-guardian2","figure"),
    Output("pa-famsup","figure"),
    Output("pa-edu-cmt","children"),
    Input("f-school","value"), Input("f-sex","value"),
    Input("f-address","value"), Input("f-age","value"),
    Input("page-store","data"),
)
def upd_parental(school,sex,address,age,page):
    if page!="parental": return [dash.no_update]*6
    d = apply_filters(school,sex,address,age)
    pstatus_fig = px.box(d, x="Pstatus_lbl", y="G3", color="Pstatus_lbl",
                         color_discrete_sequence=[G1,O1],
                         labels={"Pstatus_lbl":"Parent Status","G3":"G3"},
                         notched=True)
    pstatus_fig.update_layout(**L(title="Parent Cohabitation → G3"), showlegend=False)

    guardian_fig = d.groupby("guardian_lbl")["G3"].mean().reset_index()
    guardian_fig = px.bar(guardian_fig, x="guardian_lbl", y="G3",
                          color="guardian_lbl",
                          color_discrete_sequence=[G1,O1,B1],
                          labels={"guardian_lbl":"Guardian","G3":"Avg G3"})
    guardian_fig.update_layout(**L(title="Guardian → Avg G3"), showlegend=False)

    famsup_fig = px.box(d, x="famsup_lbl", y="G3", color="famsup_lbl",
                        color_discrete_map={"Yes":G1,"No":O1},
                        labels={"famsup_lbl":"Family Support","G3":"G3"},
                        notched=True)
    famsup_fig.update_layout(**L(title="Family Educational Support → G3"),
                             showlegend=False)

    return (fig_parent_edu(d), fig_mjob_fjob(d),
            pstatus_fig, guardian_fig, famsup_fig,
            cmt_parent_edu(d))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  14. ACADEMIC PAGE CHARTS                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("ac-studytime","figure"),       Output("ac-support","figure"),
    Output("ac-travel","figure"),          Output("ac-failures","figure"),
    Output("ac-freetime","figure"),        Output("ac-goout","figure"),
    Output("ac-internet","figure"),
    Output("ac-studytime-cmt","children"), Output("ac-failures-cmt","children"),
    Input("f-school","value"), Input("f-sex","value"),
    Input("f-address","value"), Input("f-age","value"),
    Input("page-store","data"),
)
def upd_academic(school,sex,address,age,page):
    if page!="academic": return [dash.no_update]*9
    d = apply_filters(school,sex,address,age)

    fail_fig = px.box(d, x="failures", y="G3", color="failures",
                      color_discrete_sequence=CATCL, notched=True,
                      labels={"failures":"Past Failures","G3":"G3"})
    fail_fig.add_hline(y=10, line_dash="dot", line_color=MU, line_width=1)
    fail_fig.update_layout(**L(title="Past Failures → G3"), showlegend=False)

    ft_fig = px.box(d, x="freetime", y="G3", color="freetime",
                    color_discrete_sequence=CATCL, notched=True,
                    labels={"freetime":"Free Time (1=Low→5=High)","G3":"G3"})
    ft_fig.update_layout(**L(title="Free Time After School → G3"), showlegend=False)

    go_fig = fig_goout_gograde(d)

    inet_fig = px.box(d, x="internet_lbl", y="G3", color="internet_lbl",
                      color_discrete_map={"Yes":G1,"No":O1},
                      notched=True,
                      labels={"internet_lbl":"Internet at Home","G3":"G3"})
    inet_fig.update_layout(**L(title="Internet Access → G3"), showlegend=False)

    return (fig_studytime_violin(d), fig_support_impact(d),
            fig_travel_failures(d), fail_fig, ft_fig, go_fig, inet_fig,
            cmt_studytime(d), cmt_failures(d))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  15. WELL-BEING PAGE CHARTS                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("wb-alc-heat","figure"),       Output("wb-health","figure"),
    Output("wb-goout2","figure"),         Output("wb-romantic","figure"),
    Output("wb-health2","figure"),
    Output("wb-alc-heat-cmt","children"),
    Input("f-school","value"), Input("f-sex","value"),
    Input("f-address","value"), Input("f-age","value"),
    Input("page-store","data"),
)
def upd_wellbeing(school,sex,address,age,page):
    if page!="wellbeing": return [dash.no_update]*6
    d = apply_filters(school,sex,address,age)

    rom_fig = px.box(d, x="romantic_lbl", y="G3", color="romantic_lbl",
                     color_discrete_map={"Yes":O1,"No":G1}, notched=True,
                     labels={"romantic_lbl":"Romantic Relationship","G3":"G3"})
    rom_fig.update_layout(**L(title="Romantic Relationship → G3"), showlegend=False)

    hlt_fig = px.box(d, x="health", y="G3", color="health",
                     color_discrete_sequence=SEQ5, notched=True,
                     labels={"health":"Health Status (1=Bad→5=Great)","G3":"G3"})
    hlt_fig.update_layout(**L(title="Self-Rated Health → G3"), showlegend=False)

    return (fig_alcohol_heatmap(d), fig_health_freetime(d),
            fig_goout_gograde(d), rom_fig, hlt_fig,
            cmt_alc_heat(d))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  16. GRADES PAGE CHARTS                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("gr-hist","figure"),          Output("gr-evo","figure"),
    Output("gr-trend","figure"),         Output("gr-scatter","figure"),
    Output("gr-donut","figure"),         Output("gr-corr-bars","figure"),
    Output("gr-corr-heat","figure"),
    Output("gr-hist-cmt","children"),    Output("gr-evo-cmt","children"),
    Output("gr-scatter-cmt","children"), Output("gr-corr-bars-cmt","children"),
    Input("f-school","value"), Input("f-sex","value"),
    Input("f-address","value"), Input("f-age","value"),
    Input("page-store","data"),
)
def upd_grades(school,sex,address,age,page):
    if page!="grades": return [dash.no_update]*11
    d = apply_filters(school,sex,address,age)
    return (fig_grade_hist(d), fig_grade_evolution(d),
            fig_grade_trend(d), fig_scatter_absences(d),
            fig_success_donut(d), fig_correlation_bars(d),
            fig_correlation_heatmap(d),
            cmt_grade_hist(d), cmt_grade_evo(d),
            cmt_scatter_abs(d), cmt_corr_bars(d))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  17. DATA TABLE CALLBACK                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("data-table","data"),
    Output("tbl-meta","children"),
    Output("tbl-stats","children"),
    Input("f-school","value"), Input("f-sex","value"),
    Input("f-address","value"), Input("f-age","value"),
    Input("page-store","data"),
)
def upd_table(school,sex,address,age,page):
    if page!="data":
        return dash.no_update, dash.no_update, dash.no_update
    d = apply_filters(school,sex,address,age)
    n = len(d)
    meta  = f"Student Data — {n} records selected"
    stats = (f"Min G3: {d['G3'].min():.0f}  ·  Max G3: {d['G3'].max():.0f}  ·  "
             f"Std: {d['G3'].std():.2f}  ·  Avg age: {d['age'].mean():.1f}  ·  "
             f"At-risk: {(d['G3']<10).sum()} ({(d['G3']<10).mean()*100:.1f}%)") if n else "No data"
    return d.to_dict("records"), meta, stats


# ╔══════════════════════════════════════════════════════════════════╗
# ║  17b. STUDENT PROFILE CARD CALLBACK                             ║
# ╚══════════════════════════════════════════════════════════════════╝

@app.callback(
    Output("profile-card", "children"),
    Input("data-table", "active_cell"),
    State("data-table", "derived_viewport_data"),
    State("f-school", "value"), State("f-sex", "value"),
    State("f-address", "value"), State("f-age", "value"),
    prevent_initial_call=True,
)
def show_profile(active_cell, viewport_data, school, sex, address, age):
    if not active_cell or not viewport_data:
        return dash.no_update
    row_idx = active_cell.get("row", 0)
    if row_idx >= len(viewport_data):
        return dash.no_update
    row = viewport_data[row_idx]

    rs = int(row.get("risk_score", 0))
    if rs >= 70:
        badge_cls, badge_lbl = "risk-high", "High Risk"
    elif rs >= 40:
        badge_cls, badge_lbl = "risk-med", "Medium Risk"
    else:
        badge_cls, badge_lbl = "risk-low", "Low Risk"

    success_val = str(row.get("success", ""))
    success_color = O1 if "Risk" in success_val else G1 if "Excellent" in success_val else O2

    left_rows = [
        ("School",     row.get("school_lbl", "—")),
        ("Gender",     row.get("sex_lbl", "—")),
        ("Age",        row.get("age", "—")),
        ("Address",    row.get("address_lbl", "—")),
        ("Study Time", row.get("studytime_lbl", "—")),
        ("Failures",   row.get("failures", "—")),
        ("Absences",   row.get("absences", "—")),
    ]
    right_rows = [
        ("G1 (Term 1)",    row.get("G1", "—")),
        ("G2 (Term 2)",    row.get("G2", "—")),
        ("G3 (Final)",     row.get("G3", "—")),
        ("Category",       success_val),
        ("Walc (Wknd alc)", row.get("Walc", "—")),
        ("Dalc (Work alc)", row.get("Dalc", "—")),
        ("Risk Score",     f"{rs} / 100"),
    ]

    def kv_table(rows):
        return html.Table([
            html.Tbody([
                html.Tr([
                    html.Td(k, style={"color": MU, "fontWeight": "600",
                                      "fontSize": "11px", "textTransform": "uppercase",
                                      "letterSpacing": ".5px", "paddingRight": "14px",
                                      "paddingBottom": "5px", "whiteSpace": "nowrap"}),
                    html.Td(str(v), style={"fontSize": "12px", "paddingBottom": "5px"}),
                ]) for k, v in rows
            ])
        ])

    return html.Div([
        html.Div([
            html.P("Student Profile", className="ct"),
            html.P("Profil de l'étudiant sélectionné dans la table", className="cs"),
        ], className="ch"),
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div(str(rs),
                                 style={"fontSize": "40px", "fontWeight": "700",
                                        "lineHeight": "1", "marginBottom": "4px",
                                        "color": O1 if rs >= 70 else O2 if rs >= 40 else G2}),
                        html.Div("Risk Score / 100",
                                 style={"fontSize": "10px", "color": MU,
                                        "textTransform": "uppercase", "letterSpacing": ".8px"}),
                        html.Div(html.Span(badge_lbl, className=badge_cls),
                                 style={"marginTop": "10px"}),
                        html.Div(html.Span(success_val,
                                           style={"background": success_color, "color": WH,
                                                  "padding": "4px 10px", "borderRadius": "20px",
                                                  "fontSize": "11px", "fontWeight": "600"}),
                                 style={"marginTop": "8px"}),
                    ], style={"textAlign": "center", "padding": "12px 0"}),
                ], xs=12, sm=3),
                dbc.Col(kv_table(left_rows),  xs=12, sm=4),
                dbc.Col(kv_table(right_rows), xs=12, sm=5),
            ]),
        ], style={"padding": "16px 22px 20px"}),
    ], className="cc")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  17c. AT-RISK TABLE CALLBACK                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

_RISK_COLS = ["school_lbl", "sex_lbl", "age", "address_lbl",
              "failures", "absences", "Walc", "G1", "G2", "G3", "risk_score"]

@app.callback(
    Output("atrisk-section", "children"),
    Input("f-school", "value"), Input("f-sex", "value"),
    Input("f-address", "value"), Input("f-age", "value"),
    Input("page-store", "data"),
)
def upd_atrisk(school, sex, address, age, page):
    if page != "data":
        return dash.no_update
    d = apply_filters(school, sex, address, age)
    at_risk = d[d["G3"] < 10].sort_values("risk_score", ascending=False)
    n_risk = len(at_risk)
    if n_risk == 0:
        return html.Div("No at-risk students in current selection.",
                        style={"padding": "16px 20px", "color": MU, "fontSize": "13px"})
    return html.Div([
        html.Div([
            html.P(f"At-Risk Students — {n_risk} records · sorted by risk score ↓",
                   className="ct"),
            html.P("G3 < 10 · risk_score = failures(30) + absences>10(20) + high alcohol(20) + low study(15) + no higher edu(15)",
                   className="cs"),
        ], className="ch"),
        html.Div([
            dash_table.DataTable(
                data=at_risk[[c for c in _RISK_COLS if c in at_risk.columns]].to_dict("records"),
                columns=[{"name": c, "id": c}
                         for c in _RISK_COLS if c in at_risk.columns],
                page_size=10,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "8px 12px",
                            "fontFamily": FN, "fontSize": "12px",
                            "border": f"1px solid {BD}"},
                style_header={"backgroundColor": O1, "color": WH,
                              "fontWeight": "600", "fontSize": "11px",
                              "textTransform": "uppercase"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": BG},
                    {"if": {"filter_query": "{risk_score} >= 70",
                            "column_id": "risk_score"},
                     "backgroundColor": "#fff1ee", "color": O1, "fontWeight": "700"},
                    {"if": {"filter_query": "{risk_score} >= 40 && {risk_score} < 70",
                            "column_id": "risk_score"},
                     "color": O2, "fontWeight": "600"},
                ],
            ),
        ], style={"padding": "12px 20px 20px"}),
    ], className="cc")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  18. EXPORT CALLBACKS                                           ║
# ╚══════════════════════════════════════════════════════════════════╝

def _fdf(school, sex, address, age):
    return apply_filters(school, sex, address, age)

@app.callback(
    Output("dl-csv","data"),
    Input("btn-csv","n_clicks"),
    State("f-school","value"), State("f-sex","value"),
    State("f-address","value"), State("f-age","value"),
    prevent_initial_call=True,
)
def exp_csv(_,school,sex,address,age):
    d  = _fdf(school,sex,address,age)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return dcc.send_data_frame(d.to_csv, f"student_performance_{ts}.csv", index=False)

@app.callback(
    Output("dl-excel","data"),
    Input("btn-excel","n_clicks"),
    State("f-school","value"), State("f-sex","value"),
    State("f-address","value"), State("f-age","value"),
    prevent_initial_call=True,
)
def exp_excel(_,school,sex,address,age):
    d  = _fdf(school,sex,address,age)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        d.to_excel(w, sheet_name="Student Data", index=False)
        pd.DataFrame({
            "Metric":["N","Avg G3","Failure %","Avg absences"],
            "Value": [len(d), round(d["G3"].mean(),2),
                      round((d["G3"]<10).mean()*100,1),
                      round(d["absences"].mean(),1)],
        }).to_excel(w, sheet_name="Summary", index=False)
    buf.seek(0)
    return dcc.send_bytes(buf.getvalue(), f"student_performance_{ts}.xlsx")

@app.callback(
    Output("dl-html","data"),
    Input("btn-html","n_clicks"),
    State("f-school","value"), State("f-sex","value"),
    State("f-address","value"), State("f-age","value"),
    prevent_initial_call=True,
)
def exp_html(_,school,sex,address,age):
    d  = _fdf(school,sex,address,age)
    ts_h = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ts_f = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    n = len(d)
    html_report = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Student Performance Report — {ts_h}</title>
<style>
body{{font-family:Helvetica,sans-serif;margin:48px;color:#111827;max-width:960px;}}
h1{{color:#2d6a4f;border-bottom:2px solid #e76f51;padding-bottom:8px;font-size:22px;}}
h2{{color:#2d6a4f;font-size:15px;margin-top:32px;}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:16px 0;}}
.kpi{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:16px 18px;}}
.kpi-v{{font-size:28px;font-weight:700;color:#2d6a4f;}}
.kpi-l{{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;}}
.box{{background:#f9fafb;border-left:4px solid #52b788;padding:14px 18px;border-radius:6px;margin:10px 0;}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:12px;}}
th{{background:#2d6a4f;color:#fff;padding:8px 10px;text-align:left;font-size:11px;}}
td{{padding:7px 10px;border-bottom:1px solid #e5e7eb;}}
tr:nth-child(even){{background:#f9fafb;}}
.footer{{margin-top:48px;font-size:11px;color:#9ca3af;border-top:1px solid #e5e7eb;padding-top:12px;}}
</style></head><body>
<h1>📚 Student Performance in Portugal — Analysis Report</h1>
<p><strong>Generated:</strong> {ts_h} &nbsp;|&nbsp;
   <strong>School:</strong> {school} &nbsp;|&nbsp;
   <strong>Gender:</strong> {sex} &nbsp;|&nbsp;
   <strong>Address:</strong> {address} &nbsp;|&nbsp;
   <strong>Age:</strong> {min(age)}–{max(age)}</p>

<h2>Executive Summary</h2>
<div class="grid">
  <div class="kpi"><div class="kpi-v">{n}</div><div class="kpi-l">Students</div></div>
  <div class="kpi"><div class="kpi-v">{d["G3"].mean():.1f}/20</div><div class="kpi-l">Avg G3</div></div>
  <div class="kpi"><div class="kpi-v">{(d["G3"]<10).mean()*100:.1f}%</div><div class="kpi-l">Failure Rate</div></div>
  <div class="kpi"><div class="kpi-v">{d["absences"].mean():.1f}</div><div class="kpi-l">Avg Absences</div></div>
</div>

<h2>Key Findings</h2>
<div class="box">📚 <strong>Study time &gt;5h/week</strong> → +3 points average grade vs &lt;2h/week</div>
<div class="box">🍷 <strong>High weekend alcohol (Walc 4–5)</strong> → −4 points average impact</div>
<div class="box">🏆 <strong>Zero past failures vs 1+</strong> → ≈6 point grade gap</div>
<div class="box">📖 <strong>Aspiring to higher education</strong> → +2 points average vs those who don't</div>
<div class="box">👩 <strong>Mother's education</strong> is the strongest family predictor (r≈0.22)</div>

<h2>Grade Distribution</h2>
<table><thead><tr><th>Category</th><th>Count</th><th>%</th><th>Avg G3</th></tr></thead><tbody>
{''.join([f'<tr><td>{cat}</td><td>{(d["success"]==cat).sum()}</td>'
          f'<td>{(d["success"]==cat).mean()*100:.1f}%</td>'
          f'<td>{d[d["success"]==cat]["G3"].mean():.2f}</td></tr>'
          for cat in ["At Risk  (<10)","Average  (10–13)","Excellent  (≥14)"]])}
</tbody></table>

<h2>Data Sample (first 40 rows)</h2>
{d[["school_lbl","sex_lbl","age","address_lbl","studytime_lbl",
    "failures","absences","G1","G2","G3","success"]].head(40).to_html(index=False, border=0)}

<div class="footer">Student Performance Dashboard · AMSE Mag1 2025-2026 · Zodigui Sekongo</div>
</body></html>"""
    return dcc.send_string(html_report, f"performance_report_{ts_f}.html")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  19. ENTRY POINT                                                ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 65)
    print("  Student Performance Dashboard - Portugal")
    print("  http://127.0.0.1:8051")
    print("  Dash multi-page · 7 analytical pages · 40+ charts")
    print("=" * 65)
    app.run(debug=True, port=8051, host="127.0.0.1")
