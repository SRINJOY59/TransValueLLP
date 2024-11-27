import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import os

DATA_DIR = "DATA"

criteria = [
    "4_month_rolling_return",
    "monthly_return",
    "variance_of_monthly_returns",
    "sharpe_ratio",
    "CAGR",
    "max_drawdown",
    "excess_return",
    "treynor_ratio",
    "diversification_score"
]

showing_criteria = {
    "4_month_rolling_return": "4-month rolling return",
    "monthly_return": "Monthly return",
    "variance_of_monthly_returns": "Variance of monthly returns",
    "sharpe_ratio": "Sharpe ratio",
    "CAGR": "CAGR",
    "max_drawdown": "Max drawdown",
    "excess_return": "Excess return",
    "treynor_ratio": "Treynor ratio",
    "diversification_score": "Diversification score"
}

if 'comparisons_made' not in st.session_state:
    st.session_state.comparisons_made = {}

st.title("Mutual Fund Scheme Selection using AHP")

user_id = st.text_input("Enter your unique login ID:", placeholder="e.g., user123")

st.markdown("""
### How to Use:
1. **Select up to 6 criteria** that are important for mutual fund evaluation.
2. **Compare criteria pairs:** Rate their relative importance using a slider.
3. **View results:** Calculate weights, check consistency, and rank mutual funds.
4. If the **Consistency Ratio** is too high (> 0.1), adjust your comparisons.
""")

selected_criteria = st.multiselect(
    "Select up to 6 criteria for evaluation:",
    options=criteria,
    format_func=lambda x: showing_criteria[x],
    max_selections=6
)

if selected_criteria:
    pairwise_comparisons = list(combinations(selected_criteria, 2))
    st.subheader("Pairwise Comparisons")

    for crit_1, crit_2 in pairwise_comparisons:
        st.markdown(f"**Compare {showing_criteria[crit_1]} and {showing_criteria[crit_2]}**")
        first_selected = st.radio(
            f"Which criterion you want to compare with the other?",
            options=[crit_1, crit_2],
            format_func=lambda x: showing_criteria[x],
            key=f"radio_{crit_1}_{crit_2}"
        )
        comparison_value = st.slider(
            "Rate the importance (1: Equal, 9: Extreme importance):",
            min_value=1.0,
            max_value=9.0,
            value=1.0,
            step=0.1,
            key=f"slider_{crit_1}_{crit_2}"
        )
        if first_selected == crit_1:
            st.session_state.comparisons_made[(crit_1, crit_2)] = comparison_value
            st.session_state.comparisons_made[(crit_2, crit_1)] = 1 / comparison_value
            st.success(f"{showing_criteria[crit_1]} selected as the first criterion for comparison.")
        else:
            st.session_state.comparisons_made[(crit_2, crit_1)] = comparison_value
            st.session_state.comparisons_made[(crit_1, crit_2)] = 1 / comparison_value
            st.success(f"{showing_criteria[crit_2]} selected as the first criterion for comparison.")


if st.session_state.comparisons_made:
    st.write("### Current Comparisons")
    for (crit_1, crit_2), value in st.session_state.comparisons_made.items():
        st.write(f"{showing_criteria[crit_1]} vs {showing_criteria[crit_2]}: {value:.2f}")

def calculate_ahp_weights(comparisons, selected_criteria):
    matrix = pd.DataFrame(np.eye(len(selected_criteria)), index=selected_criteria, columns=selected_criteria)
    for (crit_1, crit_2), value in comparisons.items():
        matrix.loc[crit_1, crit_2] = value
        matrix.loc[crit_2, crit_1] = 1 / value
    eigvals, eigvecs = np.linalg.eig(matrix.values)
    max_index = np.argmax(eigvals)
    priority_vector = eigvecs[:, max_index].real
    priority_vector /= priority_vector.sum()
    return priority_vector, matrix

def calculate_consistency_ratio(matrix, priority_vector):
    n = len(matrix)
    lamda_max = (np.dot(matrix.values, priority_vector) / priority_vector).mean()
    ci = (lamda_max - n) / (n - 1)
    ri_values = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45]
    ri = ri_values[n - 1] if n <= len(ri_values) else 1.5
    return ci / ri

def append_or_create_csv(data, user_id):
    file_name = f"ranked_funds_{user_id}.csv"
    file_path = file_name  
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        combined_data = pd.concat([existing_data, data]).drop_duplicates().reset_index(drop=True)
        combined_data.to_csv(file_path, index=False)
    else:
        data.to_csv(file_path, index=False)

def load_data_in_chunks(data_dir, chunksize=100000):
    chunk_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    for file in chunk_files:
        chunk = pd.read_csv(file, chunksize=chunksize)
        for data in chunk:
            yield data

if st.button("Analyse Results"):
    if st.session_state.comparisons_made and selected_criteria:
        priority_vector, matrix = calculate_ahp_weights(st.session_state.comparisons_made, selected_criteria)
        consistency_ratio = calculate_consistency_ratio(matrix, priority_vector)

        if consistency_ratio < 0.1:
            st.success(f"Consistency Ratio: {consistency_ratio:.2f} (Acceptable)")

            mutual_fund_chunks = load_data_in_chunks(DATA_DIR)
            results = []

            with st.spinner("Ranking mutual funds..."):
                for chunk in mutual_fund_chunks:
                    if set(selected_criteria).issubset(chunk.columns):
                        chunk["Score"] = chunk[selected_criteria].dot(priority_vector)
                        results.append(chunk)
                ranked_funds = pd.concat(results).sort_values(by="Score", ascending=False)

            st.write("### Top Mutual Fund Schemes")
            top_funds = ranked_funds.drop_duplicates("scheme_code").head(10)
            st.dataframe(top_funds)

            if user_id:
                append_or_create_csv(top_funds, user_id)
                st.success(f"History updated successfully for user `{user_id}`.")
            else:
                st.error("Please enter a unique identifier to save your results.")

            for scheme in top_funds["scheme_name"]:
                nav_data = ranked_funds[ranked_funds["scheme_name"] == scheme]["nav"]
                if not nav_data.empty:
                    plt.figure(figsize=(10, 6))
                    sns.kdeplot(nav_data, fill=True, color="blue", linewidth=2)
                    plt.title(f"NAV Distribution for {scheme}")
                    plt.xlabel("NAV")
                    plt.ylabel("Density")
                    st.pyplot(plt)
        else:
            st.error(f"Consistency Ratio: {consistency_ratio:.2f} (Too high! Please review your comparisons for logical consistency.)")
    else:
        st.error("Please complete all pairwise comparisons.")