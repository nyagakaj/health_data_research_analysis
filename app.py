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
    
    # ─── Tab 7: Stakeholders ────────────────────────────────────────
    with tabs[6]:
        st.header("7. Stakeholder Mapping")
        collab_cols = [c for c in df.columns if re.search(r"entity|collabor", c, re.I)]
        orgs = (
            df[collab_cols]
            .astype(str)
            .stack()
            .str.split(';')
            .explode()
            .str.strip()
            .value_counts()
            .head(20)
            .rename_axis("Organization")
            .reset_index(name="Count")
        )
        st.table(orgs)
        st.download_button(
            "Download collaborators list",
            orgs.to_csv(index=False),
            "collaborators.csv",
            "text/csv"
        )
    
    # ─── Tab 8: Policy & Legislation ───────────────────────────────
    with tabs[7]:
        st.header("8. Policy & Legislation")
        pcols = {t: [c for c in df.columns if t in c.lower()] for t in ['law','priority','gdp','budget']}
        law = (
            df.groupby('Country')[pcols['law']].first()
              .eq('Yes').any(axis=1)
              .astype(int)
              .reset_index(name='HasLaw')
        )
        pr = (
            df.groupby('Country')[pcols['priority']].first()
              .eq('Yes').any(axis=1)
              .astype(int)
              .reset_index(name='HasPriority')
        )
        gdp = pd.to_numeric(
            df.groupby('Country')[pcols['gdp'][0]].first(),
            errors='coerce'
        ).reset_index(name='GDP%')
        bud = pd.to_numeric(
            df.groupby('Country')[pcols['budget'][0]].first(),
            errors='coerce'
        ).reset_index(name='Budget%')
        pol = law.merge(pr, on='Country').merge(gdp, on='Country').merge(bud, on='Country')
        st.table(pol.set_index('Country'))
        melt8 = pol.melt(id_vars='Country', var_name='Metric', value_name='Value')
        fig8 = px.bar(
            melt8, x='Country', y='Value', color='Metric', barmode='group',
            title="Policy & Budget Metrics"
        )
        st.plotly_chart(fig8, use_container_width=True)
    
    progress.progress(100)

else:
    st.info("Please upload **both** an English and a French CSV to begin.")
