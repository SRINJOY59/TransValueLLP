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
### Instructions for Using the Analytical Hierarchy Process (AHP):

1. **Select the Criteria for Evaluation:**
   - Begin by selecting the criteria that are important for evaluating the options (in this case, mutual fund schemes).
   - These criteria will serve as the foundation for comparison and ranking.

2. **Pairwise Comparisons:**
   - A pairwise comparison matrix will be created based on the criteria you selected.
   - For each pair of criteria, you will be asked to determine which one is more important in the context of your decision-making process.
   - Rate the importance of one criterion relative to the other using a scale of 1 to 9, where:
     - **1** means both criteria are equally important.
     - **3** means one criterion is moderately more important than the other.
     - **5** means one criterion is strongly more important than the other.
     - **7** means one criterion is very strongly more important than the other.
     - **9** means one criterion is extremely more important than the other.
     - **Even values (2, 4, 6, 8)** can also be used to express intermediate levels of importance.

3. **Using the Slider for Rating Importance:**
   - As you evaluate each pair of criteria, use the slider to rate the relative importance of one over the other.
   - The slider will allow you to select a value between 1 and 9, making it easy to quantify the importance of each comparison.

4. **Calculate Weights:**
   - Once all the pairwise comparisons are made, the system will calculate the **priority vector**, which represents the relative importance (or weights) of each criterion.
   - The weights will give you an idea of how much each criterion contributes to the final decision, with higher values indicating more important criteria.

5. **Consistency Check:**
   - A key feature of AHP is ensuring that the comparisons you’ve made are **consistent**.
   - The system will calculate the **consistency ratio** to assess the logical consistency of your comparisons.
     - A consistency ratio below **0.1** is considered acceptable.
     - If the ratio is above **0.1**, it suggests that the comparisons are inconsistent, and you should review them.

6. **Ranking the Alternatives:**
   - Once the weights are calculated and consistency is checked, the mutual fund schemes (or any other alternatives you are evaluating) will be ranked based on the selected criteria and their corresponding weights.
   - The schemes will be sorted from the most favorable to the least favorable based on the weighted sum of the criteria.

7. **Visualizing the Results:**
   - You will be able to see a ranking of the mutual fund schemes along with their scores.
   - You can also view the distribution of the Net Asset Value (NAV) for each of the top-ranked schemes.

By following these steps, you will be able to make a well-informed decision using the power of the Analytical Hierarchy Process (AHP).
""")


selected_criteria = st.multiselect(
    "Select criteria for evaluation:",
    options=criteria,
    format_func=lambda x: showing_criteria[x]
)

if selected_criteria:
    pairwise_comparisons = list(combinations(selected_criteria, 2))
    st.subheader("Pairwise Comparisons")

    for crit_1, crit_2 in pairwise_comparisons:
        st.markdown(f"**Compare {showing_criteria[crit_1]} and {showing_criteria[crit_2]}**")
        first_selected = st.radio(
            f"Which criterion is more important?",
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
        else:
            st.session_state.comparisons_made[(crit_2, crit_1)] = 1 / comparison_value

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

def append_or_create_csv(data, user_id, data_dir):
    file_name = f"ranked_funds_{user_id}.csv"
    file_path = os.path.join(data_dir, file_name)
    os.makedirs(data_dir, exist_ok=True)  
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

if selected_criteria:
    st.session_state.criteria = selected_criteria  
else:
    st.session_state.criteria = []
    
if st.button("Analyse Results"):
    if st.session_state.comparisons_made and selected_criteria:
        priority_vector, matrix = calculate_ahp_weights(st.session_state.comparisons_made, selected_criteria)
        consistency_ratio = calculate_consistency_ratio(matrix, priority_vector)

        # st.write("### Pairwise Comparison Matrix")
        # st.dataframe(matrix)

        # st.write("### Calculated Weights")
        # st.dataframe(pd.DataFrame(priority_vector, index=selected_criteria, columns=["Weight"]))

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
                append_or_create_csv(top_funds, user_id, DATA_DIR)
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
            st.error(f"Consistency Ratio: {consistency_ratio:.2f} (Too high, review comparisons)")

    else:
        st.error("Please complete all pairwise comparisons.")
