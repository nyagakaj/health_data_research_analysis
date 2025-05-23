# Health Research Dashboard

This Streamlit application provides an interactive dashboard for analyzing health research site data across six West African countries (Nigeria, Togo, Ghana, Guinea‑Bissau, Gambia, Sierra Leone). It supports bilingual data ingestion (English/French), automatic translation of column headers, data normalization, and eight key objectives—each displayed in its own tab with visualizations and downloadable outputs.

## Features

1. **Data Ingestion & Preprocessing**

   * Upload English and French CSV files via the sidebar.
   * Automatic de-duplication and translation of French headers.
   * Schema alignment, removal of empty columns, and concatenation.
   * Filtering to the six target countries (handles multiple spellings of Guinea‑Bissau).
   * Normalization of boolean survey responses (`Yes`/`No`).

2. **Interactive Progress Bar**

   * Real-time sidebar progress indicator during data loading and processing steps.

3. **Tabbed Interface**

   * **Tab 1: Identification**

     * Categorize sites into Basic Science, Preclinical, Clinical Trials, Epidemiological, Other.
     * Table summary of counts by country.
     * Interactive grouped bar chart.
     * Exhaustive list of research sites with download option.
   * **Tab 2: Capacity Evaluation**

     * Computes a capability score based on identification flags.
     * Bar chart of average score per country with HTML download.
   * **Tab 3: Human Resources**

     * Percent‑yes summaries for staff sufficiency/expertise questions.
     * Average numeric staff counts.
   * **Tab 4: Translational Research**

     * Flags Phase I clinical trial reporting.
     * Bar chart of counts per country.
   * **Tab 5: Infrastructure**

     * Infrastructure index based on security, ISO, and advanced availability flags.
     * Average index bar chart.
   * **Tab 6: Ethics & Regulatory**

     * IRB/ethics presence flags.
     * Counts of in-house IRBs per country.
   * **Tab 7: Stakeholders**

     * Top collaborator organizations (parsed from semicolon‑separated entries).
     * Downloadable CSV of top collaborators.
   * **Tab 8: Policy & Legislation**

     * Flags for research laws and priority lists.
     * GDP and budget percentages.
     * Grouped bar chart of policy metrics.

4. **Modern UI**

   * Colored, large tabs with custom CSS.
   * Responsive layouts and interactive Plotly visuals.
   * Download buttons for data tables and charts.

## Requirements

* Python 3.8+
* Streamlit
* pandas
* numpy
* plotly
* deep-translator

Install dependencies:

```bash
pip install streamlit pandas numpy plotly deep-translator
```

## Running the App

1. Save your English and French CSV files locally.
2. Launch the app:

   ```bash
   streamlit run app.py
   ```
3. In the browser, upload both CSVs via the sidebar.
4. Navigate tabs to explore analyses and download outputs.

## Customization

* **Countries**: Modify the `targets` list in the code to adjust focus.
* **Tab Colors**: Update the CSS in `st.markdown` for new color themes.
* **Column Patterns**: Extend regex patterns in each tab block to capture additional survey items.

## License

This project is released under the MIT License. Feel free to adapt and distribute.
