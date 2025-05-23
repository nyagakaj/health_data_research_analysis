# Health Research Dashboard

This Streamlit application provides an interactive dashboard to analyze and visualize health research site data across selected African countries. It supports both English and French datasets, automatically aligns and normalizes data, and offers eight analytical objectives:

1. **Identification of Research Sites**
2. **Capacity Evaluation**
3. **Human Resource Assessment**
4. **Translational Research (Phase I)**
5. **Infrastructure Analysis**
6. **Ethics & Regulatory**
7. **Stakeholder Mapping**
8. **Policy & Legislation**

---

## Features

* **Dual-language support**: Upload English and/or French CSVs; French headers are translated to English.
* **Automatic schema alignment**: Merges datasets, drops empty columns, and filters to target countries.
* **Boolean normalization**: Converts various yes/no/checked values to `Yes`/`No`.
* **Interactive tabs**: One tab per objective, each with summary tables, visualizations (bar charts, sunbursts, heatmaps, etc.), and download buttons.
* **Progress indicator**: Sidebar progress bar tracks data processing steps.
* **Downloadable outputs**: CSVs and HTML for charts, as well as detailed tables.

## Prerequisites

* Python 3.8 or higher
* Pip

## Dependencies

Install the required packages via pip:

```bash
pip install streamlit pandas numpy deep-translator plotly
```

## Running the App

1. Clone or download this repository.
2. Navigate to the project directory:

   ```bash
   cd <project-folder>
   ```
3. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```
4. Open the URL provided by Streamlit (e.g., `http://localhost:8501`) in your browser.

## Usage

1. **Upload CSVs**: Use the sidebar to upload one or both of your English and French datasets (`.csv`).
2. **Process Data**: Click the **Process Data** button. A progress bar will indicate the current step.
3. **Navigate Tabs**: Explore the eight tabs for different analyses:

   * **Identification**: Overview of research site categories (Basic Science, Preclinical, Clinical Trials, Epidemiological) with bar and sunburst charts.
   * **Capacity**: Average capability scores per country, plus heatmap of site counts.
   * **Human Resources**: Percentage of “Yes” responses and average staff counts, with bar charts for each.
   * **Translational**: Counts of Phase I trial sites and scatter plot against capability.
   * **Infrastructure**: Infrastructure index distribution and average scores per country.
   * **Ethics/Reg**: IRB presence and pie charts by country.
   * **Stakeholders**: Cleaned list of collaborating organizations per country.
   * **Policy**: Site‐level policy existence, dissemination, implementation, budget allocation, SOP coverage; radar and grouped bar charts.
4. **Download Tables & Charts**: Each tab provides download buttons for CSVs and chart HTML.

## Data Format

* Input CSVs must contain a `Country` column and at least one column matching the research indicators. The app will auto-detect a site-name column (`Name of Institute` or French equivalent).
* French datasets: headers will be translated automatically.

## Customization

* **Target countries**: Modify the `targets` list in the code to include additional countries.
* **Regex patterns**: Adjust the category/indicator regex patterns under each tab as needed.
* **Visual styling**: Update the embedded CSS in `st.markdown` for tab colors and styling.

---

**Author**: John Nyagaka & Victorine Maikem
**License**: MIT
