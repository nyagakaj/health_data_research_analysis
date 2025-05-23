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

/* Tabs styling */
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

# ───── Sidebar: File Uploader & Process Button ─────────────────────────────────
st.sidebar.title("Upload CSVs")
en_file = st.sidebar.file_uploader("English dataset", type="csv")
fr_file = st.sidebar.file_uploader("French dataset", type="csv")
process = st.sidebar.button("Process Data")
progress = st.sidebar.progress(0)

# Utility: ensure unique column names
def make_unique(cols):
    cnt, out = {}, []
    for c in cols:
        cnt[c] = cnt.get(c, 0)
        out.append(c if cnt[c]==0 else f"{c}_{cnt[c]}")
        cnt[c] += 1
    return out

if process:
    # Step 1: Load & prepare
    progress.progress(10)
    dfs = []
    if en_file:
        dfs.append(pd.read_csv(en_file))
    if fr_file:
        df_fr = pd.read_csv(fr_file)
        df_fr.columns = make_unique(df_fr.columns)
        translator = GoogleTranslator(source='fr', target='en')
        cmap = {}
        for c in df_fr.columns:
            try:
                cmap[c] = translator.translate(c).strip().capitalize()
            except:
                cmap[c] = c
        df_fr.rename(columns=cmap, inplace=True)
        df_fr.columns = make_unique(df_fr.columns)
        dfs.append(df_fr)
    if not dfs:
        st.sidebar.error("Upload at least one CSV file.")
        st.stop()
    progress.progress(30)

    # Step 2: Align schemas
    all_cols = sorted(set().union(*[df.columns for df in dfs]))
    empty = pd.DataFrame(columns=all_cols)
    aligned = [pd.concat([empty, dfX], ignore_index=True)[all_cols] for dfX in dfs]
    df = pd.concat(aligned, ignore_index=True)
    dropcols = [c for c in all_cols if df[c].isna().all()]
    df.drop(columns=dropcols, inplace=True)
    progress.progress(50)

    # Step 3: Filter countries
    df['Country'] = df['Country'].astype(str) \
        .str.replace(r'Guinea[\s-]?Bissau','Guinea-Bissau',flags=re.I)
    targets = ["Nigeria","Togo","Ghana","Guinea-Bissau","Gambia","Sierra Leone"]
    df = df[df['Country'].isin(targets)].copy()
    progress.progress(60)

    # Step 4: Normalize yes/no
    def unify(v):
        if not isinstance(v,str): return v
        t=v.strip().lower()
        if t in ('oui','yes','checked','true','1'): return 'Yes'
        if t in ('non','no','unchecked','false','0'): return 'No'
        return v
    df = df.applymap(unify)
    progress.progress(70)

    # Auto-detect site-name col (English or French)
    name_col = next((c for c in df.columns if re.search(r'\bname\b',c,re.I)
                     or re.search(r'nom.*institut',c,re.I)), df.columns[0])

    # Step 5: Tabs & Analyses
    progress.progress(80)
    tabs = st.tabs(["1. Identification","2. Capacity","3. Human Resources",
                    "4. Translational","5. Infrastructure","6. Ethics/Reg",
                    "7. Stakeholders","8. Policy"])
    
    # ─── Tab 1: Identification ─────────────────────────────────
    with tabs[0]:
        st.header("1. Identification of Research Sites")
        cats = {
            "BasicScience":[r"\bbasic\b",r"fundamental"],
            "Preclinical":[r"preclinical"],
            "ClinicalTrials":[r"clinical"],
            "Epidemiological":[r"epidemiolog"]
        }
        bool_cols=[]
        for cat,pats in cats.items():
            cols=[c for c in df.columns if any(re.search(p,c,re.I) for p in pats)]
            df[f"Is{cat}"]=df[cols].eq('Yes').any(axis=1)
            bool_cols.append(f"Is{cat}")
        def list_cats(r):
            eng=[cat for cat in cats if r[f"Is{cat}"]]
            return ", ".join(eng) if eng else "Other"
        df['EngagedCategories']=df.apply(list_cats,axis=1)

        # summary table + bar
        summary1=(df.groupby('Country')[bool_cols].sum()
                    .rename(columns=lambda x:x.replace('Is','')))
        summary1['Other']=df.groupby('Country')['EngagedCategories'].apply(lambda s:(s=='Other').sum())
        st.table(summary1)
        melt1=summary1.reset_index().melt('Country',var_name='Category',value_name='Count')
        fig1=px.bar(melt1,x='Country',y='Count',color='Category',barmode='group',
                    title="Sites by Category & Country")
        st.plotly_chart(fig1,use_container_width=True)
        # additional: sunburst
        fig1b=px.sunburst(melt1,path=['Country','Category'],values='Count',
                          title='Sunburst of Research Sites by Country and Category')
        st.plotly_chart(fig1b,use_container_width=True)

        # exhaustive site list
        name_col = ('Name of Institute' if 'Name of Institute' in df.columns else
                    'Nom de l\'institut' if 'Nom de l\'institut' in df.columns else name_col)
        sites=(df[[name_col,'Country','EngagedCategories']]
               .drop_duplicates([name_col,'Country'])
               .rename(columns={name_col:'SiteName','EngagedCategories':'Categories'}))
        st.subheader("All Sites")
        st.table(sites)
        st.download_button("Download sites list (CSV)",sites.to_csv(index=False),
                           "research_sites.csv","text/csv")

    # ─── Tab 2: Capacity ─────────────────────────────────
    with tabs[1]:
        st.header("2. Capacity Evaluation")
        df['CapabilityScore']=df[bool_cols].sum(axis=1)
        cap=df.groupby('Country')['CapabilityScore'].mean().round(2).reset_index()
        fig2=px.bar(cap,x='Country',y='CapabilityScore',color='Country',
                    title="Avg Capability Score by Country",range_y=[0,len(bool_cols)])
        st.plotly_chart(fig2,use_container_width=True)
        st.download_button("Download capacity chart (HTML)",fig2.to_html(),"capacity.html")
        # additional: heatmap
        heat= pd.DataFrame(melt1[melt1['Category'].isin([c.replace('Is','') for c in bool_cols])]
                            .pivot(index='Country',columns='Category',values='Count').fillna(0))
        fig2b=px.imshow(heat,labels=dict(x="Category",y="Country",color="Count"),
                        title="Heatmap of Site Counts per Category & Country")
        st.plotly_chart(fig2b,use_container_width=True)

    # ─── Tab 3: Human Resource Assessment ────────────────────────────────────────
    with tabs[2]:
        st.header("3. Human Resource Assessment")

        # ── 1. Define high-level boolean indicator groups ───────────────────
        bool_groups = {
            "Staff Sufficiency (%)": [r"sufficient to effectively carry out"],
            "Clinical Staff Avail. (%)": [r"availability of clinical staff"],
            "Lab Staff Avail. (%)":      [r"availability of laboratory staff"],
            "Pharmacy Staff Avail. (%)": [r"availability of pharmacy staff"],
            "Bioinformatics (%)":        [r"bioinformatics"],
            "Cell Culture (%)":          [r"cell culture"],
            "Org. Synthesis (%)":        [r"organic synthesis"],
            "Virology (%)":              [r"virology"],
        }

        # Compute per-site boolean flags (1 if any matching question == Yes)
        for name, pats in bool_groups.items():
            cols = [
                c for c in df.columns
                if any(re.search(p, c, re.IGNORECASE) for p in pats)
            ]
            df[name] = (
                df[cols]
                .eq("Yes")
                .any(axis=1)
                .astype(int)
            )

        # Aggregate to country-level % Yes
        bool_summary = (
            df.groupby("Country")[list(bool_groups.keys())]
              .mean()
              .multiply(100)
              .round(1)
        )

        # ── 2. Define high-level numeric groups ─────────────────────────────
        num_groups = {
            "Num Other Staff": [r"number of other staff"],
            "Num PhD":         [r"doctorate|phd"],
            "Num MSc":         [r"master's|msc"],
            "Total Staff":     [r"total number of personnel"],
        }

        # Compute per-site numeric indicators (max across duplicates)
        for name, pats in num_groups.items():
            cols = [
                c for c in df.columns
                if any(re.search(p, c, re.IGNORECASE) for p in pats)
            ]
            if cols:
                numeric_df = df[cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
                df[name] = numeric_df.max(axis=1).fillna(0)
            else:
                df[name] = 0

        # Aggregate to country-level averages
        num_summary = (
            df.groupby("Country")[list(num_groups.keys())]
              .mean()
              .round(1)
        )

        # ── 3. Combine into one tidy table ────────────────────────────────
        combined = pd.concat([bool_summary, num_summary], axis=1)
        st.subheader("Staff Sufficiency & Average Counts by Country")
        st.table(combined)

        # ── 4. Visual #1: %-Yes bar chart ───────────────────────────────
        melt_b = (
            bool_summary
            .reset_index()
            .melt("Country", var_name="Indicator", value_name="% Yes")
        )
        fig_hr1 = px.bar(
            melt_b,
            x="Country",
            y="% Yes",
            color="Indicator",
            barmode="group",
            title="Percentage of 'Yes' Responses by Country"
        )
        st.plotly_chart(fig_hr1, use_container_width=True)

        # ── 5. Visual #2: Average counts bar chart ───────────────────────
        melt_n = (
            num_summary
            .reset_index()
            .melt("Country", var_name="Measure", value_name="Average")
        )
        fig_hr2 = px.bar(
            melt_n,
            x="Country",
            y="Average",
            color="Measure",
            barmode="group",
            title="Average Staff Counts by Country"
        )
        st.plotly_chart(fig_hr2, use_container_width=True)



    # ─── Tab 4: Translational ─────────────────────────────────
    with tabs[3]:
        st.header("4. Translational Research (Phase I)")
        trans=[c for c in df.columns if re.search(r"phase.*i",c,re.I)]
        df['HasPhaseI']=df[trans].eq('Yes').any(axis=1)
        tr=df.groupby('Country')['HasPhaseI'].sum().reset_index()
        fig4=px.bar(tr,x='Country',y='HasPhaseI',color='Country',
                    title="Sites Reporting Phase I Trials",range_y=[0,tr['HasPhaseI'].max()+1])
        st.plotly_chart(fig4,use_container_width=True)
        # additional: scatter bubble vs capability
        cap_df=cap.merge(tr,on='Country')
        fig4b=px.scatter(cap_df,x='CapabilityScore',y='HasPhaseI',size='HasPhaseI',
                         color='Country',title='Phase I Trials vs. Capability Score')
        st.plotly_chart(fig4b,use_container_width=True)

    # ─── Tab 5: Infrastructure ─────────────────────────────────
    with tabs[4]:
        st.header("5. Infrastructure Analysis")
        infra_terms=['availability of advanced','level of biosecurity','iso certification']
        infra_cols=[c for c in df.columns if any(t in c.lower() for t in infra_terms)]
        df['InfraIndex']=df[infra_cols].eq('Yes').sum(axis=1)
        infra=df.groupby('Country')['InfraIndex'].mean().round(1).reset_index()
        fig5=px.bar(infra,x='Country',y='InfraIndex',color='Country',
                    title="Avg Infrastructure Index",range_y=[0,len(infra_cols)])
        st.plotly_chart(fig5,use_container_width=True)
        # additional: violin
        fig5b=px.violin(df,x='Country',y='InfraIndex',
                         title='Distribution of Infrastructure Index by Country')
        st.plotly_chart(fig5b,use_container_width=True)

    # ─── Tab 6: Ethics & Regulatory ─────────────────────────────────
    with tabs[5]:
        st.header("6. Ethics & Regulatory")
        ethic_terms=['ethic','irb','regul','guidelines']
        ethic_cols=[c for c in df.columns if any(t in c.lower() for t in ethic_terms)]
        df['HasIRB']=df[ethic_cols].eq('Yes').any(axis=1)
        er=df.groupby('Country')['HasIRB'].sum().reset_index()
        fig6=px.bar(er,x='Country',y='HasIRB',color='Country',
                    title="Sites with In‐House IRBs")
        st.plotly_chart(fig6,use_container_width=True)
        # additional: pie facets
        pie_df=df.groupby(['Country','HasIRB']).size().reset_index(name='Count')
        fig6b=px.pie(pie_df,names='HasIRB',values='Count',facet_col='Country',
                     title='IRB Coverage by Country')
        st.plotly_chart(fig6b,use_container_width=True)

            # ─── Tab 7: Stakeholder Mapping ────────────────────────────────────────
    with tabs[6]:
        st.header("7. Stakeholder Mapping")

        # 1. Free-text columns that may list collaborators
        free_text_cols = [
            'Other (Please specify)',
            'If yes, list the research collaborations in the last 5 years'
        ]

        # 2. Extract every raw entry, keeping only non-empty site names
        records = []
        for col in free_text_cols:
            if col in df.columns:
                exploded = (
                    df[col]
                      .astype(str)
                      .str.split(r'[;,]+')
                      .explode()
                      .str.strip()
                ).dropna()
                for idx, raw in exploded.items():
                    raw = raw.strip()
                    if not raw:
                        continue
                    # coerce site to string, skip blanks
                    site_raw = df.at[idx, name_col]
                    site = str(site_raw).strip()
                    if not site or site.lower() in ('nan',):
                        continue
                    records.append({
                        'Country':       df.at[idx, 'Country'],
                        'Site':          site,
                        'RawStakeholder': raw
                    })

        site_stake_df = pd.DataFrame(records)

        # 3. Further split any numbered or starred lists into individual names
        def split_items(raw):
            tmp = re.sub(r'\d+\.', ';', raw)
            tmp = re.sub(r'\*+', ';', tmp)
            parts = re.split(r'[;,\n]+', tmp)
            return [p.strip() for p in parts if p.strip()]

        site_clean = (
            site_stake_df
              .assign(Stakeholder=site_stake_df['RawStakeholder']
                                         .apply(split_items))
              .explode('Stakeholder')
        )

        # 4. Group by country & stakeholder, concatenating only valid strings
        def join_sites(sites):
            unique = {str(x).strip() for x in sites if isinstance(x, str) and x.strip()}
            return "; ".join(sorted(unique))

        grouped = (
            site_clean
              .groupby(['Country','Stakeholder'])
              .agg(Sites=('Site', join_sites))
              .reset_index()
        )

        # 5. Display & download
        st.subheader("Stakeholders by Country")
        st.table(grouped)
        st.download_button(
            "Download cleaned stakeholders (CSV)",
            grouped.to_csv(index=False),
            "cleaned_stakeholders_by_country.csv",
            "text/csv"
        )


    # ─── Tab 8: Policy & Legislation ─────────────────────────────────────
    with tabs[7]:
        st.header("8. Policy & Legislation")

        # 1️⃣ Column names
        policy_exists_col       = 'Is there a health research policy in your country?'
        policy_disseminated_col = 'Has the policy been disseminated?'
        policy_implemented_col  = 'Is the policy currently under implementation?'
        budget_col              = "What percentage of the national health budget is allocated to health-related R&D, considering the AU's 2% target?"
        sop_cols = [
            'Available SOPs (choice=Consent Process)',
            'Available SOPs (choice=Subject Recruitment)',
            'Available SOPs (choice=Safety Management)',
            'Available SOPs (choice=Subject Retention)',
            'Available SOPs (choice=Laboratory and Sample Management)',
            'Available SOPs (choice=IP Management)',
            'Available SOPs (choice=Documentation/Filing/ Archiving)',
            'Available SOPs (choice=EC/IRB, RA Submission)',
            'Available SOPs (choice=Communication Plan)',
            'Available SOPs (choice=Site Organogram)',
            'Available SOPs (choice=Investigator Oversight and Training)',
            'Available SOPs (choice=Data Management)',
            'Available SOPs (choice=Quality Management)',
            'Available SOPs (choice=Medical Referral)',
            'Available SOPs (choice=Emergency Management)',
            'Available SOPs (choice=Finance or Budget)',
            'Available SOPs (choice=Community Engagement)',
            'Available SOPs (choice=Other (Specify))'
        ]

        # 2️⃣ Yes→1 helper
        yes_set = {'yes','oui','checked'}
        def to_bin(x):
            return 1 if str(x).strip().lower() in yes_set else 0

        # 3️⃣ Site-level flags
        df['PolicyExists']       = df[policy_exists_col].map(to_bin)
        df['PolicyDisseminated'] = df[policy_disseminated_col].map(to_bin)
        df['PolicyImplemented']  = df[policy_implemented_col].map(to_bin)

        # 4️⃣ Budget as fraction
        df['Budget_pct'] = (
            df[budget_col]
              .astype(str)
              .str.rstrip('%')
              .replace('', '0')
              .astype(float)
              .clip(0,100) / 100.0
        )

        # 5️⃣ SOP coverage
        sop_flags = df[sop_cols].applymap(to_bin)
        df['SOP_Coverage'] = sop_flags.sum(axis=1) / len(sop_cols)

        # 6️⃣ Build site_policy DataFrame
        site_policy = pd.DataFrame({
            'Country':      df['Country'],
            'Site':         df['Name of Institute'],
            'Exists':       df['PolicyExists'],
            'Disseminated': df['PolicyDisseminated'] & df['PolicyExists'],
            'Implemented':  df['PolicyImplemented']  & df['PolicyExists'],
            'Budget':       df['Budget_pct'],
            'SOP_Coverage': df['SOP_Coverage']
        })

        # 7️⃣ Country aggregates
        country_summary = (
            site_policy
              .groupby('Country')
              .agg(
                  pct_with_policy   = ('Exists',      'mean'),
                  pct_disseminated  = ('Disseminated','mean'),
                  pct_implemented   = ('Implemented', 'mean'),
                  avg_budget_alloc  = ('Budget',      'mean'),
                  avg_sop_coverage  = ('SOP_Coverage','mean'),
                  num_sites         = ('Site',        'count')
              )
              .reset_index()
        )
        # implementation gap
        country_summary['implementation_gap'] = (
            country_summary['pct_with_policy'] - country_summary['pct_implemented']
        )

        # 8️⃣ Display table (percent → “xx.x%”)
        disp = country_summary.copy()
        for p in ['pct_with_policy','pct_disseminated','pct_implemented','implementation_gap']:
            disp[p] = (disp[p]*100).round(1).astype(str) + '%'
        disp[['avg_budget_alloc','avg_sop_coverage']] = disp[['avg_budget_alloc','avg_sop_coverage']].round(2)

        st.subheader("Country-level Policy & Legislation Summary")
        st.table(disp.set_index('Country'))

        # 9️⃣ Bar chart of policy metrics
        melt = country_summary.melt(
            id_vars='Country',
            value_vars=['pct_with_policy','pct_disseminated','pct_implemented'],
            var_name='Metric',
            value_name='Value'
        )
        label_map = {
            'pct_with_policy':  '% With Policy',
            'pct_disseminated': '% Disseminated',
            'pct_implemented':  '% Implemented'
        }
        melt['Metric'] = melt['Metric'].map(label_map)

        fig8 = px.bar(
            melt,
            x='Country',
            y='Value',
            color='Metric',
            barmode='group',
            title="Policy Existence, Dissemination & Implementation by Country",
            labels={'Value':'Proportion (0–1)'}
        )
        st.plotly_chart(fig8, use_container_width=True)

        # 10️⃣ Radar chart
        melt_radar = country_summary.melt(
            id_vars='Country',
            value_vars=['pct_with_policy','pct_disseminated','pct_implemented','implementation_gap'],
            var_name='Metric',
            value_name='Value'
        )
        fig8b = px.line_polar(
            melt_radar,
            r='Value',
            theta='Metric',
            color='Country',
            line_close=True,
            title='Policy Radar Chart by Country'
        )
        st.plotly_chart(fig8b, use_container_width=True)

        # 11️⃣ Download
        st.download_button(
            "Download Policy Summary (CSV)",
            country_summary.to_csv(index=False),
            "policy_summary.csv",
            "text/csv"
        )

    progress.progress(100)

else:
    st.sidebar.info("Upload one or both CSV files, then click 'Process Data'.")