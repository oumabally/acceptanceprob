import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import sympy as sp

st.set_page_config(layout="wide")

st.title("Customer Acceptance Modeling Dashboard")
st.write("Explore how price, lead time, and size impact acceptance probability.")


# SYMBOLIC VARIABLES
lt_sym, p_sym, s_sym = sp.symbols('lt p s')
beta0_sym, beta1_sym, beta2_sym, beta3_sym, prob_max_sym = sp.symbols(
    'beta0 beta1 beta2 beta3 prob_max'
)

# MODEL SELECTION
#Drop down menu of different fucntion to model acceptance probability 

st.sidebar.subheader("Acceptance Probability Model")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic",
        "Hyperbolic Tangent",
        "Arctangent",
        "Rational Function"
    ]
)

# COMMON EXPONENT
z = ((beta3_sym * s_sym)
    *
    ( beta1_sym * lt_sym +
    beta2_sym * p_sym
    ))

# MODEL DEFINITIONS
if model_choice == "Logistic":
    expr = prob_max_sym / (
        1 + beta0_sym * sp.exp(z)
    )

elif model_choice == "Hyperbolic Tangent":
    expr = prob_max_sym * (
        (1 - sp.tanh(z)) / 2
    )

elif model_choice == "Arctangent":
    expr = prob_max_sym * (
        0.5 - (sp.atan(z) / sp.pi)
    )

elif model_choice == "Rational Function":
    expr = prob_max_sym * (
        1 / (1 + z**2)
    )

# DISPLAY EQUATION
st.sidebar.latex(sp.latex(expr))

# CREATE NUMERICAL FUNCTIONS
func = sp.lambdify(
    (
        lt_sym,
        p_sym,
        s_sym,
        beta0_sym,
        beta1_sym,
        beta2_sym,
        beta3_sym,
        prob_max_sym
    ),
    expr,
    "numpy"
)

# derivatives

dP_dlt_expr = sp.diff(expr, lt_sym)
dP_dp_expr = sp.diff(expr, p_sym)

dP_dlt_func = sp.lambdify(
    (
        lt_sym,
        p_sym,
        s_sym,
        beta0_sym,
        beta1_sym,
        beta2_sym,
        beta3_sym,
        prob_max_sym
    ),
    dP_dlt_expr,
    "numpy"
)

dP_dp_func = sp.lambdify(
    (
        lt_sym,
        p_sym,
        s_sym,
        beta0_sym,
        beta1_sym,
        beta2_sym,
        beta3_sym,
        prob_max_sym
    ),
    dP_dp_expr,
    "numpy"
)

# =========================
# SIDEBAR PARAMETERS
# =========================
st.sidebar.header("Model Parameters")

beta0 = st.sidebar.slider("β₀", 0.1, 5.0, 0.5)
beta3 = st.sidebar.slider("β₃", 0.1, 5.0, 1.0)
prob_max = st.sidebar.slider("Max Acceptance Probability", 0.1, 1.0, 0.85)

st.sidebar.header("Operational Inputs")
sizes = st.sidebar.multiselect(
    "Sizes", [1,2,3,4], default=[1,2])

lead_times = st.sidebar.multiselect(
    "Lead Times", [1,2,3,4,5], default=[1,2])

baseline_costs = st.sidebar.multiselect(
    "Costs", list(range(1,11)), default=[1,2,3])

markups = st.sidebar.multiselect(
    "Markups", [0,0.1,0.15,0.2,0.25,0.3,0.4,0.5],
    default=[0.1,0.2])

if not sizes or not lead_times or not baseline_costs or not markups:
    st.warning("Please select at least one value for all inputs.")
    st.stop()

#Normalized components in terms of user input
max_lt = max(lead_times)
max_size = max(sizes)
min_size = min(sizes)
max_price = max(
    c * (1 + m)
    for c in baseline_costs
    for m in markups)

min_lt = min(lead_times)
min_price = min(
    c * (1 + m)
    for c in baseline_costs
    for m in markups
)

# Elasticies calculated for the new models 
def price_elasticity(lead_time, size, markup, baseline_cost, beta0, beta1, beta2, beta3, prob_max):
    price = baseline_cost * (1 + markup)
    #normlaization of variables 
    lt_norm = (
    (lead_time - min_lt)
    / (max_lt - min_lt)
)
    p_norm = (
    (price - min_price)
    / (max_price - min_price)
)
    s_norm = 0.25 + 0.75 * (1 - size / max_size)

    P = func(lt_norm, p_norm, s_norm, beta0, beta1, beta2, beta3, prob_max)
    dP_dp = dP_dp_func(lt_norm, p_norm, s_norm, beta0, beta1, beta2, beta3,prob_max)

    elasticity = dP_dp * ((price / max_price) / P)

    return elasticity if P != 0 else 0

def leadtime_elasticity(lead_time, size, markup, baseline_cost, beta0, beta1, beta2, beta3, prob_max):
    price = baseline_cost * (1 + markup)
    #normlaization of variables 
    lt_norm = (
    (lead_time - min_lt)
    / (max_lt - min_lt)
)
    p_norm = (
    (price - min_price)
    / (max_price - min_price)
)
    s_norm = 0.25 + 0.75 * (1 - size / max_size)

    P = func(lt_norm, p_norm, s_norm, beta0, beta1, beta2, beta3, prob_max)
    dP_dlt = dP_dlt_func(lt_norm, p_norm, s_norm, beta0, beta1, beta2, beta3,prob_max)

    elasticity = dP_dlt * ((lead_time / max_lt) / P)

    return elasticity if P != 0 else 0


def plot_price_heatmap(beta1_range, beta2_range, sizes, costs, lead_times, markups, beta0, beta3, prob_max):
    results = []
    for b1 in beta1_range:
        for b2 in beta2_range:
            e_vals = []
            for s in sizes:
                for c in costs:
                    for lt in lead_times:
                        for m in markups:
                            e_vals.append(abs(price_elasticity(
                                lt, s, m, c, beta0, b1, b2, beta3, prob_max)))

            results.append({
                'Beta 1': round(b1, 2),
                'Beta 2': round(b2, 2),
                'Elasticity': np.mean(e_vals),
                'Min': min(e_vals),
                'Max': max(e_vals)
            })

    df = pd.DataFrame(results)

    matrix_df = df.pivot(index='Beta 2', columns='Beta 1', values='Elasticity')
    matrix_df = matrix_df.sort_index(ascending=False)

    cmap = ListedColormap(['#ffff00', '#ffa500', '#ff0000'])

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={'label': 'Price Elasticity Magnitude'},
        vmin=0, vmax=1.5,
        ax=ax
    )

    ax.set_title("Price Elasticity Heat Map", fontsize=14)
    ax.set_xlabel("β₁ (Lead Time Sensitivity)")
    ax.set_ylabel("β₂ (Price Sensitivity)")

    return fig

def plot_leadtime_heatmap(beta1_range, beta2_range, sizes, costs, lead_times, markups, beta0, beta3, prob_max):

    results = []

    for b1 in beta1_range:
        for b2 in beta2_range:
            lte_vals = []

            for s in sizes:
                for c in costs:
                    for lt in lead_times:
                        for m in markups:
                            lte_vals.append(abs(leadtime_elasticity(
                                lt, s, m, c, beta0, b1, b2, beta3, prob_max
                            )))

            results.append({
                'Beta 1': round(b1, 2),
                'Beta 2': round(b2, 2),
                'Elasticity': np.mean(lte_vals)
            })

    df = pd.DataFrame(results)

    matrix_df = df.pivot(index='Beta 2', columns='Beta 1', values='Elasticity')
    matrix_df = matrix_df.sort_index(ascending=False)

    cmap = ListedColormap(['#e0f3f8', '#abd9e9', '#4575b4'])

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={'label': 'Lead Time Elasticity Magnitude'},
        vmin=0, vmax=1.5,
        ax=ax
    )

    ax.set_title("Lead Time Elasticity Heat Map", fontsize=14)
    ax.set_xlabel("β₁ (Lead Time Sensitivity)")
    ax.set_ylabel("β₂ (Price Sensitivity)")

    return fig


def get_combined_state(p_val, lt_val):
    #FIND intersection of elasticities
    p_cat = 0 if p_val < 0.5 else (1 if p_val <= 1.0 else 2)
    lt_cat = 0 if lt_val < 0.5 else (1 if lt_val <= 1.0 else 2)
    return (lt_cat * 3) + p_cat + 1

def plot_superimposed_heatmap(beta1_range, beta2_range, sizes, costs, lead_times, markups, beta0, beta3, prob_max):

    results = []

    for b1 in beta1_range:
        for b2 in beta2_range:

            p_vals, lt_vals = [], []

            for s in sizes:
                for c in costs:
                    for lt in lead_times:
                        for m in markups:

                            p_vals.append(abs(price_elasticity(
                                lt, s, m, c, beta0, b1, b2, beta3, prob_max
                            )))

                            lt_vals.append(abs(leadtime_elasticity(
                                lt, s, m, c, beta0, b1, b2, beta3, prob_max
                            )))

            p_avg, lt_avg = np.mean(p_vals), np.mean(lt_vals)
            state = get_combined_state(p_avg, lt_avg)

            results.append({
                'Beta 1': round(b1, 2),
                'Beta 2': round(b2, 2),
                'State': state
            })

    df = pd.DataFrame(results)

    matrix = df.pivot(index='Beta 2', columns='Beta 1', values='State')
    matrix = matrix.sort_index(ascending=False)

    colors = [
        '#e8e1b0', '#83af70', '#2d5e45',
        '#fee0d2', '#fc9272', '#de2d26',
        '#d8daeb', '#b2abd2', '#5e3c99'
    ]

    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        matrix,
        cmap=cmap,
        linewidths=.5,
        cbar=False,
        annot=True,
        ax=ax
    )

    legend_labels = {
        'LT Dominant (Strong)': '#2d5e45',
        'Price Dominant (Strong)': '#de2d26',
        'Overlap (Both Strong)': '#5e3c99'
    }

    patches = [mpatches.Patch(color=v, label=k) for k, v in legend_labels.items()]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title("Superimposed Sensitivity", fontsize=14)
    ax.set_xlabel("β₁ (Lead Time)")
    ax.set_ylabel("β₂ (Price)")

    return fig

# =========================
# HEATMAP FUNCTIONS (FIXED: USE MARKUP)
# =========================


def calc_acceptance_prob(lead_time, size, markup, baseline_cost, beta0, beta1,beta2,beta3,prob_max):

    price = baseline_cost * (1 + markup)

    # normalization
    lt_norm = (
    (lead_time - min_lt)
    / (max_lt - min_lt)
)
    p_norm = (
    (price - min_price)
    / (max_price - min_price)
)
    s_norm = 0.25 + 0.75 * (1 - size / max_size)

    return func(lt_norm, p_norm, s_norm, beta0, beta1, beta2, beta3, prob_max
                )
def plot_acceptance_heatmap(size, beta0, beta1,beta2,beta3,prob_max):
    results = []
    for lt in lead_times:
        for c in baseline_costs:
            for m in markups:
                prob = calc_acceptance_prob(lt, size, m, c, beta0, beta1, beta2, beta3, prob_max)

                price = c * (1 + m)

                results.append({
                    "Price": round(price, 2),
                    "Lead Time": lt,
                    "P": prob
                })

    df = pd.DataFrame(results)
    pivot = df.pivot(
        index="Lead Time",
        columns="Price",
        values="P")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(
        pivot,
        cmap="magma",
        vmin=0,
        vmax=prob_max,
        annot=True,
        fmt=".2f",
        ax=ax
    )

    ax.set_title(f"{model_choice} Acceptance Heatmap (Size={size})")
    ax.set_xlabel("Price (with Markup)")
    ax.set_ylabel("Lead Time")

    return fig


# =========================
# TAB 2 (FOCUS FIX)
# =========================

def plot_leadtime_curve(
    fixed_price,
    beta0,
    beta1,
    beta2,
    beta3,
    prob_max
):

    fig, ax = plt.subplots(figsize=(7,4))

    lt_vals = np.linspace(
        min(lead_times),
        max(lead_times),
        25
    )

    for s in sizes:

        probs = []

        for lt in lt_vals:

            lt_norm = (
                (lt - min_lt)
                / (max_lt - min_lt)
            )

            p_norm = (
                (fixed_price - min_price)
                / (max_price - min_price)
            )

            s_norm = 0.25 + 0.75 * (1 - s/ max_size)

            P = func(
                lt_norm,
                p_norm,
                s_norm,
                beta0,
                beta1,
                beta2,
                beta3,
                prob_max
            )

            probs.append(P)

        ax.plot(
            lt_vals,
            probs,
            marker='o',
            label=f"Size {s}"
        )

    ax.set_title(
        f"{model_choice}: Acceptance vs Lead Time"
    )

    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Acceptance Probability")

    ax.set_ylim(0, prob_max)

    ax.grid(True, alpha=0.3)

    ax.legend()

    return fig

def plot_price_curve(
    fixed_lt,
    beta0,
    beta1,
    beta2,
    beta3,
    prob_max
):

    fig, ax = plt.subplots(figsize=(7,4))

    price_vals = np.linspace(
        min(baseline_costs),
        max_price,
        25
    )

    for s in sizes:

        probs = []

        for price in price_vals:

            lt_norm = (
                (fixed_lt - min_lt)
                / (max_lt - min_lt))

            p_norm = (
                (price - min_price)
                / (max_price - min_price))

            s_norm = 0.25 + 0.75 * (1 - s / max_size)

            P = func(
                lt_norm,
                p_norm,
                s_norm,
                beta0,
                beta1,
                beta2,
                beta3,
                prob_max
            )

            probs.append(P)

        ax.plot(
            price_vals,
            probs,
            marker='o',
            label=f"Size {s}"
        )

    ax.set_title(
        f"{model_choice}: Acceptance vs Price"
    )

    ax.set_xlabel("Final Customer Price")
    ax.set_ylabel("Acceptance Probability")

    ax.set_ylim(0, prob_max)

    ax.grid(True, alpha=0.3)

    ax.legend()

    return fig

# =========================
# TABS
# =========================

tab1, tab2 = st.tabs(["Elasticity Analysis", "Acceptance Probability"])

with tab1:

    st.header("Elasticity Analysis")

    st.subheader("Beta Range Controls")

    b1_min, b1_max = st.slider("β₁ Range", 0.1, 10.0, (0.5, 6.0))
    b2_min, b2_max = st.slider("β₂ Range", 0.1, 10.0, (0.5, 6.0))
    resolution = st.slider("Bin Range", 5, 25, 12)

    beta1_range = np.linspace(b1_min, b1_max, resolution)
    beta2_range = np.linspace(b2_min, b2_max, resolution)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Elasticity")
        fig_price = plot_price_heatmap(
            beta1_range,
            beta2_range,
            sizes,
            baseline_costs,
            lead_times,
            markups,
            beta0,
            beta3,
            prob_max
        )
        st.pyplot(fig_price)

    with col2:
        st.subheader("Lead Time Elasticity")
        fig_lt = plot_leadtime_heatmap(
            beta1_range,
            beta2_range,
            sizes,
            baseline_costs,
            lead_times,
            markups,
            beta0,
            beta3,
            prob_max
        )
        st.pyplot(fig_lt)

#Superimposed Graph
    st.subheader("Superimposed Plot of Price and Leadtime Elasticity")
    fig_combined = plot_superimposed_heatmap(
    beta1_range,
    beta2_range,
    sizes,
    baseline_costs,
    lead_times,
    markups,
    beta0,
    beta3,
    prob_max
)
    st.pyplot(fig_combined)


with tab2:

    st.header("Acceptance Probability Analysis")

    beta1_single = st.slider("β₁ (Tab 2)", 0.1, 10.0, 3.0, key="beta1_tab2")
    beta2_single = st.slider("β₂ (Tab 2)", 0.1, 10.0, 3.0, key="beta2_tab2")

    col1, col2 = st.columns(2)

    # 3D PLOT (NOW USES MARKUPS)
    with col1:
        st.subheader("3D Acceptance Surface")

        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')

        for s in sizes:

            X, Y = np.meshgrid(lead_times, baseline_costs)
            Z = np.zeros_like(X, dtype=float)

            for i, c in enumerate(baseline_costs):
                for j, lt in enumerate(lead_times):

                    # use average markup for surface
                    m = np.mean(markups)

                    Z[i,j] = calc_acceptance_prob(
                        lt, s, m, c,
                        beta0, beta1_single, beta2_single, beta3, prob_max
                    )

            ax.plot_surface(X, Y, Z, alpha=0.6)

        ax.set_xlabel("Lead Time")
        ax.set_ylabel("Baseline Cost")
        ax.set_zlabel("Acceptance Probability")

        st.pyplot(fig)

    # HEATMAPS (NOW USE MARKUPS)
    with col2:
        st.subheader("Acceptance Heatmaps")

        for s in sizes:
            fig_hm = plot_acceptance_heatmap(
                s,
                beta0,
                beta1_single,
                beta2_single,
                beta3,
                prob_max
            )
            st.pyplot(fig_hm)
    
    # =========================
    # 2D ACCEPTANCE CURVES
    # =========================

    st.subheader("2D Acceptance Probability Curves")

    # Controls
    fixed_lt = st.selectbox(
        "Fixed Lead Time",
        lead_times,
        key="fixed_lt"
    )

    fixed_price = st.slider(
        "Fixed Customer Price",
        float(min(baseline_costs)),
        float(max_price),
        float(min(baseline_costs))
    )

    curve_col1, curve_col2 = st.columns(2)

    # -------------------------
    # Lead Time Curve
    # -------------------------

    with curve_col1:

        fig_lt_curve = plot_leadtime_curve(
            fixed_price,
            beta0,
            beta1_single,
            beta2_single,
            beta3,
            prob_max
        )

        st.pyplot(fig_lt_curve)

    # -------------------------
    # Price Curve
    # -------------------------

    with curve_col2:

        fig_price_curve = plot_price_curve(
            fixed_lt,
            beta0,
            beta1_single,
            beta2_single,
            beta3,
            prob_max
        )

        st.pyplot(fig_price_curve)
