import streamlit as st
import pandas as pd
import numpy as np
import re
from deep_translator import GoogleTranslator
import plotly.express as px

# ───── Page & Theme Setup ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Research Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for big, coloured tabs + progress bar
st.markdown("""
<style>
/* Progress bar colour */
.stProgress > div > div > div > div { background-color: #4caf50; }

/* Tabs: bigger font, min-width, no wrap */
.stTabs [role="tab"] {
  font-size: 18px !important;
  padding: 12px 20px !important;
  min-width: 180px;
  white-space: nowrap;
  border-radius: 8px 8px 0 0;
  margin-right: 4px;
}
/* Individual tab backgrounds */
.stTabs [role="tab"]:nth-child(1) { background: #FFCDD2; }
.stTabs [role="tab"]:nth-child(2) { background: #F8BBD0; }
.stTabs [role="tab"]:nth-child(3) { background: #BBDEFB; }
.stTabs [role="tab"]:nth-child(4) { background: #C8E6C9; }
.stTabs [role="tab"]:nth-child(5) { background: #FFECB3; }
.stTabs [role="tab"]:nth-child(6) { background: #D1C4E9; }
.stTabs [role="tab"]:nth-child(7) { background: #B2EBF2; }
.stTabs [role="tab"]:nth-child(8) { background: #FFE0B2; }
/* Selected tab darker */
.stTabs [role="tab"][aria-selected="true"]:nth-child(1) { background: #E57373 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(2) { background: #F06292 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(3) { background: #64B5F6 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(4) { background: #81C784 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(5) { background: #FFD54F !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(6) { background: #9575CD !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(7) { background: #4DD0E1 !important; color:white !important; }
.stTabs [role="tab"][aria-selected="true"]:nth-child(8) { background: #FFB74D !important; color:white !important; }
</style>
""", unsafe_allow_html=True)


# ───── Sidebar: File Uploader & Progress ────────────────────────────────────────
st.sidebar.title("Upload CSVs")
en_file = st.sidebar.file_uploader("English dataset", type="csv")
fr_file = st.sidebar.file_uploader("French dataset", type="csv")
progress = st.sidebar.progress(0)


def make_unique(cols):
    cnt, out = {}, []
    for c in cols:
        cnt[c] = cnt.get(c, 0)
        out.append(c if cnt[c] == 0 else f"{c}_{cnt[c]}")
        cnt[c] += 1
    return out


if en_file and fr_file:
    # 1️⃣ Load files
    progress.progress(10)
    df_en = pd.read_csv(en_file)
    df_fr = pd.read_csv(fr_file)
    
    # 2️⃣ Translate & dedupe French headers
    progress.progress(20)
    df_fr.columns = make_unique(df_fr.columns)
    translator = GoogleTranslator(source='fr', target='en')
    cmap = {}
    for c in df_fr.columns:
        try:
            translated = translator.translate(c)
            cmap[c] = translated.strip().capitalize()
        except:
            cmap[c] = c
    df_fr.rename(columns=cmap, inplace=True)
    df_fr.columns = make_unique(df_fr.columns)
    
    # 3️⃣ Align schemas, drop all-empty cols, concat
    progress.progress(35)
    all_cols = sorted(set(df_en.columns) | set(df_fr.columns))
    empty = pd.DataFrame(columns=all_cols)
    df_en = pd.concat([empty, df_en], ignore_index=True)[all_cols]
    df_fr = pd.concat([empty, df_fr], ignore_index=True)[all_cols]
    both_empty = [c for c in all_cols 
                  if df_en[c].isna().all() and df_fr[c].isna().all()]
    df_en.drop(columns=both_empty, inplace=True)
    df_fr.drop(columns=both_empty, inplace=True)
    df = pd.concat([df_en, df_fr], ignore_index=True)
    
    # 4️⃣ Filter to our six countries (handle “Guinea Bissau” variants)
    progress.progress(50)
    df['Country'] = df['Country'].astype(str)\
        .str.replace(r'Guinea[\s-]?Bissau', 'Guinea-Bissau', flags=re.I)
    targets = ["Nigeria","Togo","Ghana","Guinea-Bissau","Gambia","Sierra Leone"]
    df = df[df['Country'].isin(targets)].copy()
    
    # 5️⃣ Normalize yes/no answers across languages
    progress.progress(60)
    def unify_bool(v):
        if not isinstance(v, str):
            return v
        t = v.strip().lower()
        if t in ('oui','yes','checked','true','1'):
            return 'Yes'
        if t in ('non','no','unchecked','false','0'):
            return 'No'
        return v
    df = df.applymap(unify_bool)
    
    # Auto-detect site‐name column
    name_col = next((c for c in df.columns if re.search(r'\bname\b', c, re.I)),
                    df.columns[0])
    
    # 6️⃣ Set up Tabs
    progress.progress(70)
    tabs = st.tabs([
        "1. Identification", 
        "2. Capacity", 
        "3. Human Resources", 
        "4. Translational", 
        "5. Infrastructure", 
        "6. Ethics/Reg", 
        "7. Stakeholders", 
        "8. Policy"
    ])
    
    # ─── Tab 1: Identification ─────────────────────────────────────────
    with tabs[0]:
        st.header("1. Identification of Research Sites")
        cats = {
            "BasicScience":    [r"\bbasic\b", r"fundamental"],
            "Preclinical":     [r"preclinical"],
            "ClinicalTrials":  [r"clinical"],
            "Epidemiological": [r"epidemiolog"]
        }
        bool_cols = []
        for cat, pats in cats.items():
            cols = [c for c in df.columns if any(re.search(p, c, re.I) for p in pats)]
            df[f"Is{cat}"] = df[cols].eq('Yes').any(axis=1)
            bool_cols.append(f"Is{cat}")
        df['IsOther'] = ~df[bool_cols].any(axis=1)
        
        summary1 = (
            df.groupby('Country')[bool_cols + ['IsOther']]
              .sum()
              .rename(columns=lambda x: x.replace('Is',''))
        )
        st.table(summary1)
        
        melt1 = summary1.reset_index().melt('Country', var_name='Category', value_name='Count')
        fig1 = px.bar(
            melt1, x='Country', y='Count', color='Category', barmode='group',
            title="Sites by Category & Country"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Exhaustive sites list
        sites = df[[name_col, 'Country'] + bool_cols]
        sites.columns = ['SiteName', 'Country'] + list(cats.keys())
        st.subheader("All Sites")
        st.table(sites)
        st.download_button(
            "Download sites list (CSV)", 
            sites.to_csv(index=False), 
            "research_sites.csv", 
            "text/csv"
        )
    
    # ─── Tab 2: Capacity Evaluation ───────────────────────────────────
    with tabs[1]:
        st.header("2. Capacity Evaluation")
        df['CapabilityScore'] = df[bool_cols].sum(axis=1)
        cap = df.groupby('Country')['CapabilityScore'].mean().round(2).reset_index()
        fig2 = px.bar(
            cap, x='Country', y='CapabilityScore', color='Country',
            title="Avg Capability Score by Country", range_y=[0, len(bool_cols)]
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.download_button(
            "Download capacity chart (HTML)", 
            fig2.to_html(), 
            "capacity.html"
        )
    
    # ─── Tab 3: Human Resources ─────────────────────────────────────
    with tabs[2]:
        st.header("3. Human Resource Assessment")
        hr_cols = [
            'Are the number of staff sufficient to effectively carry out the duties?',
            'Are the number of staff sufficient to effectively carry out the duties?.1',
            'Are the number of staff sufficient to effectively carry out the duties?.2',
            'Are the number of staff sufficient to effectively carry out the duties?.3',
            'Are the number of staff sufficient to effectively carry out the duties?.4',
            'Are the number of staff sufficient to effectively carry out the duties?.5',
            'Availability of Clinical Staff ',
            'Availability of Laboratory Staff and total number:',
            'Availability of Pharmacy Staff and total number',
            'Availability of laboratory staff and total number:',
            'Availability of pharmacy staff and total number',
            'Does the laboratory have an expertise in bioinformatics, including chemoinformatics, genomics and transcriptomics?',
            'Does the laboratory have expertise in bioinformatics analysis, including cheminformatics, genomics, and transcriptomics?',
            'Does the laboratory have expertise in cell culture techniques, including primary cell culture and cell line development?',
            'Does the laboratory have expertise in cell culture techniques, including primary cell culture and the development of cell lines?',
            'Does the laboratory have expertise in organic synthesis, including multi-step synthesis and the ability to synthesize complex molecules?',
            'Does the laboratory have expertise in organic synthesis, including synthesis in several stages and the ability to synthesize complex molecules?',
            'Does the laboratory have expertise in virology, including propagation, titration and characterization of viruses?',
            'Does the laboratory have expertise in virology, including virus propagation, titration, and characterization?',
            'Is access to refrigerator content limited to pharmacy staff only?',
            'Is access to the pharmacy reserved for authorized staff?',
            "Is access to the refrigerator's contents limited to pharmacy staff only?  ",
            'Number of Other Staff', 'Number of Other Staff.1', 'Number of Other Staff.2',
            'Number of Other Staff:', 'Number of doctorate holders (phd) .12',
            'Number of doctorate holders (phd) .3','Number of doctorate holders (phd) .4',
            'Number of doctorate holders (phd) .5',"Number of master's holders (msc) .12",
            "Number of master's holders (msc) .3","Number of master's holders (msc) .4",
            "Number of master's holders (msc) .5",'Number of other staff members',
            'Number of other staff members:','Number of other staff.',
            'Number of other staff. 2','Number of staff with a Doctorate (PhD) ',
            'Staff Composition: Total Number of personnel '
        ]
        hr_exists = [c for c in hr_cols if c in df.columns]
        hr_bool = [c for c in hr_exists if df[c].dtype == object]
        hr_num  = [c for c in hr_exists if pd.api.types.is_numeric_dtype(df[c])]
        
        bool_summary = (
            df[hr_bool].eq('Yes')
                     .join(df['Country'])
                     .groupby('Country')
                     .mean()
                     .multiply(100)
                     .round(1)
        )
        st.subheader("Staff Sufficiency & Expertise (%)")
        st.table(bool_summary)
        
        if hr_num:
            num_summary = (
                df[hr_num]
                  .join(df['Country'])
                  .groupby('Country')
                  .mean()
                  .round(1)
            )
            st.subheader("Average Staff Counts")
            st.table(num_summary)
    
    # ─── Tab 4: Translational Research ────────────────────────────────
    with tabs[3]:
        st.header("4. Translational Research (Phase I)")
        trans_cols = [c for c in df.columns if re.search(r"phase.*i", c, re.I)]
        df['HasPhaseI'] = df[trans_cols].eq('Yes').any(axis=1)
        tr = df.groupby('Country')['HasPhaseI'].sum().reset_index()
        fig4 = px.bar(
            tr, x='Country', y='HasPhaseI', color='Country',
            title="Sites Reporting Phase I Trials", range_y=[0, tr['HasPhaseI'].max()+1]
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # ─── Tab 5: Infrastructure ───────────────────────────────────────
    with tabs[4]:
        st.header("5. Infrastructure Analysis")
        infra_terms = ['availability of advanced','level of biosecurity','iso certification']
        infra_cols = [c for c in df.columns if any(t in c.lower() for t in infra_terms)]
        df['InfraIndex'] = df[infra_cols].eq('Yes').sum(axis=1)
        infra = df.groupby('Country')['InfraIndex'].mean().round(1).reset_index()
        fig5 = px.bar(
            infra, x='Country', y='InfraIndex', color='Country',
            title="Avg Infrastructure Index", range_y=[0, len(infra_cols)]
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # ─── Tab 6: Ethics & Reg ────────────────────────────────────────
    with tabs[5]:
        st.header("6. Ethics & Regulatory")
        ethic_terms = ['ethic','irb','regul','guidelines']
        ethic_cols = [c for c in df.columns if any(t in c.lower() for t in ethic_terms)]
        df['HasIRB'] = df[ethic_cols].eq('Yes').any(axis=1)
        er = df.groupby('Country')['HasIRB'].sum().reset_index()
        fig6 = px.bar(
            er, x='Country', y='HasIRB', color='Country',
            title="Sites with In‐House IRBs"
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    # ─── Tab 7: Stakeholder Mapping ────────────────────────────────────────
    with tabs[6]:
        st.header("7. Stakeholder Mapping")

        # define which survey questions we’re using
        stakeholder_info = {
            'Research Collaborations': {
                'indicator': "National, regional, international research and continental collaborations",
                'detail':    "If yes, list the research collaborations in the last 5 years"
            }
            # you could add more categories here if needed
        }

        yes_vals = {'yes','oui','checked','true','1'}
        rows = []
        # for each category, scan every row
        for category, cols in stakeholder_info.items():
            ind_col = cols['indicator']
            det_col = cols['detail']
            if ind_col not in df.columns or det_col not in df.columns:
                continue
            for _, r in df.iterrows():
                if str(r.get(ind_col,"")).strip().lower() in yes_vals and str(r.get(det_col,"")).strip():
                    parts = re.split(r'[;,\\n]+', r[det_col])
                    for p in parts:
                        p = p.strip()
                        if p:
                            rows.append({
                                'Country':      r['Country'],
                                'Site':         r[name_col],
                                'Category':     category,
                                'Stakeholder':  p
                            })

        # build a DataFrame and drop duplicates
        stake_df = pd.DataFrame(rows).drop_duplicates()

        if stake_df.empty:
            st.info("No stakeholders found under the specified questions.")
        else:
            # pivot to wide form: one row per Site, columns for each category
            pivot = (
                stake_df
                  .groupby(['Country','Site','Category'])['Stakeholder']
                  .apply(lambda lst: "; ".join(sorted(set(lst))))
                  .unstack(fill_value="")
                  .reset_index()
            )

            st.subheader("Stakeholders by Site & Category")
            st.table(pivot)

            st.download_button(
                "Download stakeholders list (CSV)",
                pivot.to_csv(index=False),
                "stakeholders_by_site.csv",
                "text/csv"
            )

            # optional: top‐20 stakeholder frequency bar chart
            top = (
                stake_df['Stakeholder']
                .value_counts()
                .head(20)
                .reset_index()
                .rename(columns={'index':'Stakeholder','Stakeholder':'Count'})
            )
            fig7 = px.bar(
                top, x='Count', y='Stakeholder', orientation='h',
                title="Top 20 Stakeholders Across All Sites",
                labels={'Count':'Mentions','Stakeholder':''}
            )
            st.plotly_chart(fig7, use_container_width=True)

    
        # ─── Tab 8: Policy & Legislation ───────────────────────────────
    with tabs[7]:
        st.header("8. Policy & Legislation")

        # 1. Identify policy & legislation columns by keyword
        existence_cols = [
            c for c in df.columns
            if re.search(r'\b(policy exists|legislation)\b', c, re.IGNORECASE)
        ]
        implementation_cols = [
            c for c in df.columns
            if re.search(r'\b(implementation|monitoring)\b', c, re.IGNORECASE)
        ]
        budget_cols = [
            c for c in df.columns
            if re.search(r'percentage of the national health budget', c, re.IGNORECASE)
        ]
        sop_cols = [
            c for c in df.columns
            if re.search(r'available sops', c, re.IGNORECASE)
        ]

        # 2. Harmonize yes/no to binary
        yes_vals = {'yes', 'oui', 'checked'}
        def to_binary(v):
            return 1 if str(v).strip().lower() in yes_vals else 0

        # 3. Compute per‐site sub‐scores 0–1
        scores = pd.DataFrame(index=df.index)
        # existence
        if existence_cols:
            scores['Existence'] = df[existence_cols].applymap(to_binary).max(axis=1)
        else:
            scores['Existence'] = 0
        # implementation
        if implementation_cols:
            scores['Implementation'] = df[implementation_cols].applymap(to_binary).max(axis=1)
        else:
            scores['Implementation'] = 0
        # budget
        if budget_cols:
            pct = df[budget_cols[0]].astype(str).str.rstrip('%') \
                     .replace('', '0').astype(float).fillna(0)
            scores['Budget'] = pct.clip(0,100) / 100
        else:
            scores['Budget'] = 0
        # SOP coverage
        if sop_cols:
            sop_binary = df[sop_cols].applymap(lambda v: 1 if str(v).strip().lower()=='checked' else 0)
            scores['SOP_Coverage'] = sop_binary.sum(axis=1) / len(sop_cols)
        else:
            scores['SOP_Coverage'] = 0

        # 4. Composite policy & capacity score (equal weights)
        weights = {
            'Existence':      0.25,
            'Implementation': 0.25,
            'Budget':         0.25,
            'SOP_Coverage':   0.25
        }
        scores['Policy_Capacity'] = sum(scores[k] * w for k,w in weights.items())

        # 5. Attach back to df
        df = pd.concat([df, scores], axis=1)

        # 6. Country‐level summary
        country_summary = (
            df
            .groupby('Country')
            .agg(
                Existence       = ('Existence',      'mean'),
                Implementation  = ('Implementation', 'mean'),
                Budget          = ('Budget',         'mean'),
                SOP_Coverage    = ('SOP_Coverage',   'mean'),
                Policy_Capacity = ('Policy_Capacity','mean'),
                Num_Sites       = ('Country',        'count')
            )
            .round(3)
            .reset_index()
        )

        st.subheader("Country‐level Policy & Legislation Summary")
        st.table(country_summary.set_index('Country'))

        # 7. Visualization
        melt8 = country_summary.melt(
            id_vars='Country',
            value_vars=['Existence','Implementation','Budget','SOP_Coverage','Policy_Capacity'],
            var_name='Metric',
            value_name='Value'
        )
        fig8 = px.bar(
            melt8,
            x='Country',
            y='Value',
            color='Metric',
            barmode='group',
            title="Policy & Legislation Metrics by Country",
            labels={'Value':'Score (0–1)'}
        )
        st.plotly_chart(fig8, use_container_width=True)

        # 8. Download summary
        st.download_button(
            "Download Policy Summary (CSV)",
            country_summary.to_csv(index=False),
            "policy_summary.csv",
            "text/csv"
        )
