import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
    "4_month_rolling_return": "4 month rolling return",
    "monthly_return": "monthly return",
    "variance_of_monthly_returns": "variance of monthly returns",
    "sharpe_ratio": "sharpe ratio",
    "CAGR": "CAGR",
    "max_drawdown": "max drawdown",
    "excess_return": "excess return",
    "treynor_ratio": "treynor ratio",
    "diversification_score": "diversification score"
}

relevant_comparisons = [
    ("4_month_rolling_return", "monthly_return"),
    ("variance_of_monthly_returns", "sharpe_ratio"),
    ("CAGR", "excess_return"),
    ("max_drawdown", "variance_of_monthly_returns"),
    ("monthly_return", "sharpe_ratio"),
    ("CAGR", "4_month_rolling_return"),
    ("max_drawdown", "CAGR"),
    ("treynor_ratio", "4_month_rolling_return"),
    ("variance_of_monthly_returns", "monthly_return"),
    ("diversification_score", "CAGR")
]

comparison_names = [
    "4 month rolling return vs monthly return",
    "variance of monthly returns vs sharpe ratio",
    "CAGR vs excess return",
    "max drawdown vs variance of monthly returns",
    "monthly return vs sharpe ratio",
    "CAGR vs 4 month rolling return",
    "max drawdown vs CAGR",
    "treynor ratio vs 4 month rolling return",
    "variance of monthly returns vs monthly return",
    "diversification score vs CAGR"
]

mapping = {
    relevant_comparisons[0]: comparison_names[0],
    relevant_comparisons[1]: comparison_names[1],
    relevant_comparisons[2]: comparison_names[2],
    relevant_comparisons[3]: comparison_names[3],
    relevant_comparisons[4]: comparison_names[4],
    relevant_comparisons[5]: comparison_names[5],
    relevant_comparisons[6]: comparison_names[6],
    relevant_comparisons[7]: comparison_names[7],
    relevant_comparisons[8]: comparison_names[8],
    relevant_comparisons[9]: comparison_names[9]
}

if 'comparisons_made' not in st.session_state:
    st.session_state.comparisons_made = [(comp, 1.0) for comp in relevant_comparisons]

if 'selected_comparison' not in st.session_state:
    st.session_state.selected_comparison = comparison_names[0]

st.title("Mutual Fund Scheme Selection using AHP")

st.write("""
    **Rate the Importance of Criteria:**
    
    The slider is designed to help you assess how much more important one criterion is compared to another. The scale ranges from 1 to 9, where:

    - **1**: Indicates that both criteria are of equal importance.
    - **2**: Indicates that the first criterion is slightly more important than the second.
    - **4**: Indicates a moderate difference in importance, with the first criterion being notably more important.
    - **6**: Indicates a strong preference for the first criterion over the second.
    - **8**: Indicates a very strong preference, where the first criterion is much more important.
    - **9**: Indicates an absolute importance, where the first criterion is considered infinitely more important than the second.

    **Instructions:**
    - Select the criterion you want to compare from the dropdown menu.
    - Use the slider to select a value that best represents how much more important the selected criterion is compared to the other criterion.
    - Submit your comparison to record the value.

""")

from io import BytesIO


def load_guide_document():
    with open("AHP_Mutual_Fund_Scheme_Selection_Guide_Complete.docx", "rb") as file:
        return file.read()

st.sidebar.title("Download Guide")
guide_data = load_guide_document()
st.sidebar.download_button(
    label="Download Guide",
    data=guide_data,
    file_name="AHP_Mutual_Fund_Scheme_Selection_Guide_Complete.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

def create_comparison_slider(criteria_1, criteria_2):
    st.header(f"Compare {mapping[(criteria_1, criteria_2)]}")
    comparison_value = st.slider(
        f"How much more important is **{showing_criteria[criteria_1]}** compared to **{showing_criteria[criteria_2]}** for your scheme selection?", 
        1.0, 9.0, value=st.session_state.get('comparison_value', 1.0)
    )
    return comparison_value

selected_comparison = st.selectbox("Select a comparison to evaluate", options=comparison_names)
st.session_state.selected_comparison = selected_comparison

selected_index = comparison_names.index(st.session_state.selected_comparison)
crit_1, crit_2 = relevant_comparisons[selected_index]

comparison_value = create_comparison_slider(crit_1, crit_2)

if st.button("Submit Comparison"):
    comparison_tuple = (crit_1, crit_2)
    comparison_dict = {tuple(c): v for c, v in st.session_state.comparisons_made}
    
    # Update or add the comparison value
    comparison_dict[comparison_tuple] = comparison_value
    st.session_state.comparisons_made = list(comparison_dict.items())

def display_comparisons():
    comparison_df = pd.DataFrame(st.session_state.comparisons_made, columns=["Comparison", "Value"])

    for i, row in comparison_df.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            comp_pair = row["Comparison"]
            display_name = mapping.get(comp_pair, "Unknown comparison")
            st.write(display_name)
        with col2:
            st.write(row["Value"])
        with col3:
            if st.button("Delete", key=f"delete_{i}"):
                st.session_state.comparisons_made.pop(i)

display_comparisons()

def calculate_ahp_weights(comparisons):
    matrix = pd.DataFrame(np.eye(len(criteria)), index=criteria, columns=criteria)
    
    for (crit_1, crit_2), value in comparisons:
        if crit_1 in matrix.index and crit_2 in matrix.columns:
            matrix.loc[crit_1, crit_2] = value
            matrix.loc[crit_2, crit_1] = 1 / value
    
    st.write("Pairwise Comparison Matrix:")
    st.dataframe(matrix)

    if matrix.isna().values.any() or np.isinf(matrix.values).any():
        st.error("The comparison matrix contains NaN or Inf values. Please check the inputs.")
        return None, None
    
    eigvals, eigvecs = np.linalg.eig(matrix.values)
    max_index = np.argmax(eigvals)
    priority_vector = eigvecs[:, max_index]
    priority_vector = np.real(priority_vector)
    priority_vector = priority_vector / priority_vector.sum()
    
    return priority_vector, matrix

def calculate_consistency_ratio(matrix, priority_vector):
    n = len(matrix)
    lamda_max = (np.dot(matrix.values, priority_vector) / priority_vector).mean()
    ci = (lamda_max - n) / (n - 1)
    ri = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45][n - 1]
    cr = ci / ri
    return cr

def load_data_in_chunks(data_dir, chunksize=100000):
    """Loads mutual fund data in chunks from CSV files in the DATA directory."""
    chunk_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    for chunk_file in chunk_files:
        chunk = pd.read_csv(chunk_file, chunksize=chunksize)
        for data in chunk:
            yield data

if st.button("Calculate Weights and Check Consistency"):
    priority_vector, matrix = calculate_ahp_weights(st.session_state.comparisons_made)
    
    if priority_vector is not None and matrix is not None:
        consistency_ratio = calculate_consistency_ratio(matrix, priority_vector)
        
        st.write("Calculated Weights:")
        st.dataframe(pd.DataFrame(priority_vector, index=criteria, columns=["Weight"]))
        
        if consistency_ratio < 0.1:
            mutual_funds_chunks = load_data_in_chunks(DATA_DIR)
            all_results = []
            
            for mutual_funds in mutual_funds_chunks:
                scores = mutual_funds[criteria].dot(priority_vector)
                mutual_funds["Score"] = scores
                all_results.append(mutual_funds)
            
            ranked_funds = pd.concat(all_results).sort_values(by="Score", ascending=False)
            
            st.success(f"The consistency ratio is acceptable ({consistency_ratio:.2f}).")
            
            st.write("Top Mutual Fund Schemes:")
            showing_funds = ranked_funds.drop_duplicates(subset="scheme_code").head(10)
            st.dataframe(showing_funds)
            
            for scheme_name in showing_funds["scheme_name"]:
                scheme_nav_data = ranked_funds[ranked_funds["scheme_name"] == scheme_name]["nav"]
                
                plt.figure(figsize=(20, 8))  
                sns.kdeplot(scheme_nav_data, fill=True, color="darkviolet", linewidth=2)
                plt.title(f"Distribution of NAV values for {scheme_name}", fontsize=14)
                plt.xlabel("NAV", fontsize=12)
                plt.ylabel("Density", fontsize=12)
                plt.xticks(rotation=45)
                st.pyplot(plt)
                plt.clf()
                
        else:
            st.error(f"The consistency ratio is too high ({consistency_ratio:.2f}). Please review your comparisons.")
