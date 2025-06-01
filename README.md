````
# Health Research Dashboard

This Streamlit application allows users to upload one or two CSV datasets (English and/or French) containing research site information, and then explore various summaries, visualizations, and a spatial overview of core metrics across selected African countries.

## Features

1. **Landing (Upload) Page**
   - Modern, centered upload form (styled via custom CSS).
   - Two file-upload widgets: “English Dataset” and “French Dataset” (CSV only).
   - “Analyze Data” button that becomes active once at least one CSV is provided.

2. **Results Page**
   - Progress bar during data loading and preprocessing.
   - Top navigation bar with Africa CDC logo and link.
   - **Deep-Dive Configuration** (multiselect in the main panel):
     - Choose one or more countries (Nigeria, Togo, Ghana, Guinea-Bissau, Gambia, Sierra Leone).
     - If no country is selected, a prompt reminds you to select at least one.
   - Data is harmonized (French → English column names, Yes/No normalization, country filtering).

3. **Tab Layout (10 Tabs)**
   - **1. Identification**  
     - Table: counts of Basic Science, Preclinical, Clinical Trials, Epidemiological, and “Other” sites by country.  
     - Bar chart + Sunburst chart showing site counts by category and country.

   - **2. Capacity**  
     - Bar chart of average “Capability Score” per country (sum of category flags).  
     - Heatmap showing per-country counts for each identification category.

   - **3. Human Resources**  
     - Table: counts of “Yes” for each HR indicator (Clinical Staff, Lab Staff, Pharmacy Staff, Bioinformatics, Cell Culture, Organic Synthesis, Virology), plus numeric counts (Other Staff, PhD, MSc) and “Total Staff” by country.  
     - Bar charts: “Yes” counts by indicator and staff counts by country.

   - **4. Translational (Phase I)**  
     - Bar chart: count of Phase I trial sites by country.  
     - Scatter plot: relationship between average Capability Score and number of Phase I sites.

   - **5. Infrastructure**  
     - Bar chart: average “Infrastructure Index” by country.  
     - Violin chart: distribution of “InfraIndex” across sites within each country.

   - **6. Ethics & Regulatory**  
     - Bar chart: number of in-house IRB sites by country.  
     - Pie charts (faceted by country): IRB coverage (Yes/No).

   - **7. Stakeholder Mapping**  
     - Interactive table: all stakeholders per country, with “SitesList” (semicolon-separated) and “CountSites.”  
     - Download button (CSV).  
     - Table + bar chart: top 5 stakeholders across all countries.

   - **8. Policy & Legislation**  
     - Table: for each country, percentage with a policy, percentage disseminated, percentage implemented, average budget allocation (as decimal), average SOP coverage.  
     - Bar chart: “% With Policy,” “% Disseminated,” “% Implemented” by country.  
     - Pie charts + Radar chart: visual breakdown (proportions).  
     - Download button (CSV).

   - **9. Deep-Dive**  
     - If no country(s) selected: prompt to select.  
     - **Single-Country View**  
       - Bar chart: identification breakdown.  
       - “Avg Capability” metric.  
       - Bar chart: HR “Yes” counts.  
       - “Phase I Sites” metric.  
       - Bar chart: InfraIndex values.  
       - “In-house IRB Sites” metric.  
       - **Stakeholders**: table with Stakeholder, CountSites, SitesList.  
       - Metrics: Policy Exists (count), Avg Budget (%), Avg SOP Coverage (%).  
     - **Multi-Country Comparison**  
       - DataFrame: identification counts by country.  
       - Bar chart: comparative capacity scores.  
       - DataFrame: HR counts by country.  
       - DataFrame: Phase I sites by country.  
       - DataFrame: Avg InfraIndex by country.  
       - DataFrame: IRB sites by country.  
       - Table: stakeholders across selected countries.  
       - Table: policy comparison (Exists (count), Avg Budget (%), Avg SOP Coverage (%)).

   - **10. Maps**  
     - For each core metric (Avg Capability, Phase I Sites, Avg InfraIndex, IRB Sites, % With Policy), a full-width choropleth map of Africa.  
     - Each map shows countries shaded by metric value.  
     - Download button (HTML) for each map.

## Installation

1. **Clone this repository** (or copy `app.py` into your project folder).

2. **Create a virtual environment** (recommended) and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
````

The `requirements.txt` should include (at minimum):

```
pandas==2.2.3
numpy==2.2.6
plotly==6.1.2
country_converter==1.3
pycountry==24.6.1
streamlit==1.45.1
```

3. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

4. **Open** the URL printed in your terminal (usually `http://localhost:8501`).

## Usage

1. **Upload one or both CSVs** on the landing page.

   * English dataset should have column names in English.
   * French dataset column names will be mapped to English (based on first file’s headers).
   * The app only retains rows where `Country` is one of:
     `Nigeria`, `Togo`, `Ghana`, `Guinea-Bissau`, `Gambia`, `Sierra Leone`.

2. Click **Analyze Data**. You’ll be taken to the **Results** page, which displays a progress bar as data is loaded.

3. **Deep-Dive Configuration** (at top of Results):

   * Select one or more countries to filter the “Deep-Dive” tab.
   * If none is selected, the Deep-Dive tab prompts you to choose.

4. **Navigate through the tabs** (Identification, Capacity, etc.) to view tables and charts.

   * You can download CSV summaries or HTML maps via the provided buttons.

## File Structure

* `app.py`       — Main Streamlit script (upload form, results page, tabs).
* `requirements.txt` — Pin versions for all dependencies.
* `README.md`    — This documentation.

## Notes

* The upload form uses Streamlit’s built-in form styling (`.stForm`), matching the new CSS.
* All data manipulations (harmonizing Yes/No, mapping French headers, filtering countries) occur only once per session; results are cached in `st.session_state["df_full"]`.
* The “Deep-Dive” multiselect resides on the Results page, above the tabs, so that users can immediately see how selecting one or more countries affects the “Deep-Dive” content.
* Choropleth maps are generated per metric and are downloadable as standalone HTML (Plotly).

Feel free to adjust colors, add/remove target countries, or customize any visualization or CSS as needed. Enjoy exploring health research capacity across these African countries!

```
```
