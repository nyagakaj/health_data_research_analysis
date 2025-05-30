import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px

# ── Page & Theme Setup ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Health Research Dashboard", layout="wide")

# ── Routing State ──────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "upload"

# ── Top Navigation Bar ─────────────────────────────────────────────────────────
nav_html = """
<div style="position:sticky; top:0; left:0; width:100%; padding:10px 20px; z-index:1000; background: #fff;">
  <div style="display:flex; align-items:center;">
    <img src="https://impact.africacdc.org/themes/custom/thinkmodular/assets/img/africa-cdc.png"
         style="height:70px;" alt="Logo"/>
    <div style="margin-left:auto; font-size:1rem;">
      <a href="https://africacdc.org" target="_blank"
         style="color:black; text-decoration:none; font-weight:400;">
        Africa CDC Website
      </a>
    </div>
  </div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tabs styling */
.stTabs [role="tab"] {
  font-size: 18px !important;
  padding: 10px 16px !important;
  border-radius: 8px 8px 0 0;
  margin-right: 4px;
}
.stTabs [role="tab"]:nth-child(1) { background: #1A5632; }
.stTabs [role="tab"]:nth-child(2) { background: #9F2241; }
.stTabs [role="tab"]:nth-child(3) { background: #B4A269; }
.stTabs [role="tab"]:nth-child(4) { background: #348F41; }
.stTabs [role="tab"]:nth-child(5) { background: #58595B; }
.stTabs [role="tab"]:nth-child(6) { background: #9F2241; }
.stTabs [role="tab"]:nth-child(7) { background: #B4A269; }
.stTabs [role="tab"]:nth-child(8) { background: #1A5632; }
.stTabs [role="tab"][aria-selected="true"] {
  color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

#Custom STYLING FOR THE FORM - upload container
st.markdown("""
<style>
/* Widen the Streamlit form container */
.stForm {
  max-width: 1200px !important;   /* was 360px */
  width: 100% !important;          /* allow it to grow on larger screens */
  margin: 2rem auto !important;
  padding: 2rem !important;
  background: #ffffff !important;
  border-radius: 1rem !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.1) !important;
}

/* Form header */
.stForm h2 {
  font-size: 1.75rem !important;
  margin-bottom: 1rem !important;
  color: #1A5632 !important;
}

/* Uploader dropzones */
.stForm .stFileUploader > div {
  background: #f7f7f7 !important;
  border: 1px solid #ddd !important;
  border-radius: 0.5rem !important;
  padding: 1rem !important;
  transition: all 0.2s ease !important;
}
.stForm .stFileUploader > div:hover {
  border-color: #1A5632 !important;
  box-shadow: 0 0 0 4px rgba(26,86,50,0.15) !important;
}

/* Submit button */
.stForm button[type="submit"] {
  background-color: #1A5632 !important;
  color: #fff !important;
  border: none !important;
  border-radius: 0.75rem !important;
  padding: 0.75rem 1rem !important;
  font-size: 1.1rem !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
  width: 100% !important;
  margin-top: 1rem !important;
  transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stForm button[type="submit"]:hover {
  background-color: #12502e !important;
  box-shadow: 0 6px 16px rgba(0,0,0,0.15) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Shared palette ─────────────────────────────────────────────────────────────
palette = [
    "#1A5632", "#9F2241", "#B4A269", "#348F41",
    "#58595B", "#9F2241", "#B4A269", "#1A5632"
]

# ── Utility: ensure unique column names ────────────────────────────────────────
def make_unique(cols):
    cnt, out = {}, []
    for c in cols:
        cnt[c] = cnt.get(c, 0)
        out.append(c if cnt[c] == 0 else f"{c}_{cnt[c]}")
        cnt[c] += 1
    return out

# UPLOAD PAGE 
def show_upload():
    # wrap the form in our upload-form container
    st.markdown('<div class="upload-form">', unsafe_allow_html=True)
    with st.form("upload_form"):
        st.markdown('<h2>Africa Research Sites Mapping Dashboard</h2>', unsafe_allow_html=True)
        st.markdown('<p class="caption">Upload one or both CSV files to get started.</p>', unsafe_allow_html=True)

        st.markdown('<div class="dataset-boxes">', unsafe_allow_html=True)
        # English dataset box
        st.markdown('<div class="dataset-box"><h4>English Dataset</h4>', unsafe_allow_html=True)
        st.file_uploader("", type="csv", key="en_file")
        st.markdown('</div>', unsafe_allow_html=True)
        # French dataset box
        st.markdown('<div class="dataset-box"><h4>French Dataset</h4>', unsafe_allow_html=True)
        st.file_uploader("", type="csv", key="fr_file")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # submit
        if st.form_submit_button("Analyze Data"):
            if not st.session_state.get("en_file") and not st.session_state.get("fr_file"):
                st.error("Please upload at least one CSV file.")
            else:
                st.session_state.page = "results"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── RESULTS PAGE ───────────────────────────────────────────────────────────────
def show_results():
    st.markdown("### Results")
    if st.button("← Back to Upload"):
        st.session_state.page = "upload"
        st.rerun()

    en_file = st.session_state.get("en_file")
    fr_file = st.session_state.get("fr_file")
    progress = st.progress(0)

    # 1: Load
    progress.progress(10)
    df_en = pd.read_csv(en_file, keep_default_na=False) if en_file else pd.DataFrame()
    df_fr = pd.read_csv(fr_file, keep_default_na=False) if fr_file else pd.DataFrame()
    if df_en.empty and df_fr.empty:
        st.error("No data to process.")
        return

    # 2: Header map
    if not df_en.empty and not df_fr.empty:
        df_fr.rename(columns=dict(zip(df_fr.columns, df_en.columns)), inplace=True)

    # 3: Harmonize Yes/No
    yes_no_map = {
        'oui':'Yes','non':'No','yes':'Yes','no':'No',
        'checked':'Checked','coché':'Checked',
        'unchecked':'Unchecked','non coché':'Unchecked'
    }
    def harmonize(x):
        return yes_no_map.get(x.strip().lower(), x) if isinstance(x, str) else x
    if not df_fr.empty:
        df_fr = df_fr.applymap(harmonize)
    progress.progress(50)

    # 4: Align & concat
    all_cols = list(dict.fromkeys(df_en.columns.tolist() + df_fr.columns.tolist()))
    df_en = df_en.reindex(columns=all_cols, fill_value="")
    df_fr = df_fr.reindex(columns=all_cols, fill_value="")
    df = pd.concat([df_en, df_fr], ignore_index=True)
    blank_cols = [c for c in df.columns if (df[c] == "").all()]
    df.drop(columns=blank_cols, inplace=True)
    progress.progress(60)

    # 5: Filter & unify country
    df['Country'] = df['Country'].astype(str).str.replace(
        r'Guinea[\s-]?Bissau', 'Guinea-Bissau', flags=re.I
    )
    targets = ["Nigeria","Togo","Ghana","Guinea-Bissau","Gambia","Sierra Leone"]
    df = df[df['Country'].isin(targets)].copy()
    def unify(v):
        if not isinstance(v, str): return v
        t = v.strip().lower()
        if t in ('oui','yes','checked','true','1'): return 'Yes'
        if t in ('non','no','unchecked','false','0'): return 'No'
        return v
    df = df.applymap(unify)
    progress.progress(70)

    # 6: Detect name column
    name_col = next(
        (c for c in df.columns if re.search(r'\bname\b', c, re.I)
             or re.search(r'nom.*institut', c, re.I)),
        df.columns[0]
    )
    progress.progress(80)

    # 7: Tabs
    tabs = st.tabs([
        "1. Identification","2. Capacity","3. Human Resources",
        "4. Translational","5. Infrastructure","6. Ethics/Reg",
        "7. Stakeholders","8. Policy"
    ])

    # ── Tab 1: Identification ───────────────────────────────────────────────────
    with tabs[0]:
        st.header("1. Identification of Research Sites")
        cats = {
            "BasicScience":[r"\bbasic\b",r"fundamental"],
            "Preclinical":[r"preclinical"],
            "ClinicalTrials":[r"clinical"],
            "Epidemiological":[r"epidemiolog"]
        }
        bool_cols = []
        for cat, pats in cats.items():
            cols = [c for c in df.columns if any(re.search(p, c, re.I) for p in pats)]
            df[f"Is{cat}"] = df[cols].eq('Yes').any(axis=1)
            bool_cols.append(f"Is{cat}")
        def list_cats(r):
            eng = [cat for cat in cats if r[f"Is{cat}"]]
            return ", ".join(eng) if eng else "Other"
        df['EngagedCategories'] = df.apply(list_cats, axis=1)

        summary1 = df.groupby('Country')[bool_cols].sum().rename(columns=lambda x: x.replace('Is',''))
        summary1['Other'] = df.groupby('Country')['EngagedCategories'].apply(lambda s: (s=='Other').sum())
        st.table(summary1)

        melt1 = summary1.reset_index().melt('Country', var_name='Category', value_name='Count')
        fig1 = px.bar(
            melt1, x='Country', y='Count', color='Category', barmode='group',
            title="Sites by Category & Country", color_discrete_sequence=palette
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig1b = px.sunburst(
            melt1, path=['Country','Category'], values='Count',
            title='Sunburst of Research Sites by Country and Category',
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig1b, use_container_width=True)

        site_col_final = (
            'Name of Institute' if 'Name of Institute' in df.columns
            else 'Nom de l\'institut' if 'Nom de l\'institut' in df.columns
            else name_col
        )
        sites = df[[site_col_final,'Country','EngagedCategories']].drop_duplicates([site_col_final,'Country'])
        sites = sites.rename(columns={site_col_final:'SiteName','EngagedCategories':'Categories'})
        st.subheader("All Sites")
        st.table(sites)
        st.download_button("Download sites list (CSV)", sites.to_csv(index=False),
                           "research_sites.csv","text/csv")

    # ── Tab 2: Capacity ─────────────────────────────────────────────────────────
    with tabs[1]:
        st.header("2. Capacity Evaluation")
        df['CapabilityScore'] = df[bool_cols].sum(axis=1)
        cap = df.groupby('Country')['CapabilityScore'].mean().round(2).reset_index()
        fig2 = px.bar(
            cap, x='Country', y='CapabilityScore', color='Country',
            title="Avg Capability Score by Country", color_discrete_sequence=palette,
            range_y=[0,len(bool_cols)]
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.download_button("Download capacity chart (HTML)", fig2.to_html(), "capacity.html")

        heat = (
            melt1[melt1['Category'].isin([c.replace('Is','') for c in bool_cols])]
            .pivot(index='Country', columns='Category', values='Count').fillna(0)
        )
        fig2b = px.imshow(
            heat, labels=dict(x="Category", y="Country", color="Count"),
            title="Heatmap of Site Counts per Category & Country",
            color_continuous_scale=["#D0E8D8","#1A5632"],
            zmin=0, zmax=heat.values.max()
        )
        fig2b.update_layout(height=500, margin=dict(t=50,b=50))
        st.plotly_chart(fig2b, use_container_width=True)
     # Tab 3: Human Resources 
    with tabs[2]:
        st.header("3. Human Resource Assessment")
        bool_groups = {
            "Clinical Staff":[r"availability of clinical staff"],
            "Lab Staff":[r"availability of laboratory staff"],
            "Pharmacy Staff":[r"availability of pharmacy staff"],
            "Bioinformatics":[r"bioinformatics"],
            "Cell Culture":[r"cell culture"],
            "Org. Synthesis":[r"organic synthesis"],
            "Virology":[r"virology"],
        }
        for name, pats in bool_groups.items():
            cols = [c for c in df.columns if any(re.search(p,c,re.I) for p in pats)]
            df[name] = df[cols].eq("Yes").any(axis=1).astype(int)
        bool_summary = df.groupby("Country")[list(bool_groups.keys())].sum()

        num_groups = {
            "Other Staff":[r"number of other staff"],
            "PhD":[r"doctorate|phd"],
            "MSc":[r"master's|msc"],
        }
        for name, pats in num_groups.items():
            cols = [c for c in df.columns if any(re.search(p,c,re.I) for p in pats)]
            if cols:
                numeric_df = df[cols].apply(pd.to_numeric, errors="coerce")
                df[name] = numeric_df.max(axis=1).fillna(0).astype(int)
            else:
                df[name] = 0
        num_summary = df.groupby('Country')[list(num_groups.keys())].sum()
        total = bool_summary.add(num_summary, fill_value=0).sum(axis=1).astype(int)
        num_summary["Total Staff"] = total

        combined = pd.concat([bool_summary, num_summary], axis=1)
        combined.index.name = "Country"
        st.subheader("Number of “Yes” Responses & Staff Counts by Country")
        st.table(combined)

        melt_bool = bool_summary.reset_index().melt("Country", var_name="Indicator", value_name="Count of Yes")
        fig_bool = px.bar(
            melt_bool, x="Country", y="Count of Yes", color="Indicator",
            barmode="group", title="Sites Reporting “Yes” by Indicator",
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig_bool, use_container_width=True)

        melt_num = num_summary.reset_index().melt("Country", var_name="Staff Category", value_name="Count")
        fig_num = px.bar(
            melt_num, x="Country", y="Count", color="Staff Category",
            barmode="group", title="Staff Counts by Country (incl. Total Staff)",
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig_num, use_container_width=True)

    # ── Tab 4: Translational ───────────────────────────────────────────────────
    with tabs[3]:
        st.header("4. Translational Research (Phase I)")
        trans = [c for c in df.columns if re.search(r"phase.*i", c, re.I)]
        df['HasPhaseI'] = df[trans].eq('Yes').any(axis=1)
        tr = df.groupby('Country')['HasPhaseI'].sum().reset_index()
        fig4 = px.bar(
            tr, x='Country', y='HasPhaseI', color='Country',
            title="Sites Reporting Phase I Trials",
            color_discrete_sequence=palette,
            range_y=[0,tr['HasPhaseI'].max()+1]
        )
        st.plotly_chart(fig4, use_container_width=True)

        cap_df = cap.merge(tr, on='Country')
        fig4b = px.scatter(
            cap_df, x='CapabilityScore', y='HasPhaseI', size='HasPhaseI',
            color='Country', title='Phase I Trials vs. Capability Score',
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig4b, use_container_width=True)

    # ── Tab 5: Infrastructure ─────────────────────────────────────────────────
    with tabs[4]:
        st.header("5. Infrastructure Analysis")
        infra_terms=['availability of advanced','level of biosecurity','iso certification']
        infra_cols=[c for c in df.columns if any(t in c.lower() for t in infra_terms)]
        df['InfraIndex'] = df[infra_cols].eq('Yes').sum(axis=1)
        infra_df = df.groupby('Country')['InfraIndex'].mean().round(1).reset_index()
        fig5 = px.bar(
            infra_df, x='Country', y='InfraIndex', color='Country',
            title="Avg Infrastructure Index", color_discrete_sequence=palette,
            range_y=[0,len(infra_cols)]
        )
        st.plotly_chart(fig5, use_container_width=True)
        fig5b = px.violin(
            df, x='Country', y='InfraIndex',
            title='Distribution of Infrastructure Index by Country',
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig5b, use_container_width=True)

    # ── Tab 6: Ethics & Regulatory ──────────────────────────────────────────────
    with tabs[5]:
        st.header("6. Ethics & Regulatory")
        ethic_terms=['ethic','irb','regul','guidelines']
        ethic_cols=[c for c in df.columns if any(t in c.lower() for t in ethic_terms)]
        df['HasIRB'] = df[ethic_cols].eq('Yes').any(axis=1)
        er = df.groupby('Country')['HasIRB'].sum().reset_index()
        fig6 = px.bar(
            er, x='Country', y='HasIRB', color='Country',
            title="Sites with In-House IRBs", color_discrete_sequence=palette
        )
        st.plotly_chart(fig6, use_container_width=True)

        pie_df = df.groupby(['Country','HasIRB']).size().reset_index(name='Count')
        fig6b = px.pie(
            pie_df, names='HasIRB', values='Count', facet_col='Country',
            title='IRB Coverage by Country', color_discrete_sequence=palette
        )
        st.plotly_chart(fig6b, use_container_width=True)

    # ── Tab 7: Stakeholder Mapping ──────────────────────────────────────────────
    with tabs[6]:
        st.header("7. Stakeholder Mapping")
        free_cols = [
            'Other (Please specify)',
            'If yes, list the research collaborations in the last 5 years'
        ]
        records = []
        for col in free_cols:
            if col in df.columns:
                exploded = df[col].astype(str).str.split(r'[;,]+').explode().str.strip().dropna()
                for idx, raw in exploded.items():
                    if not raw or raw.lower()=="nan": continue
                    site = str(df.at[idx, name_col]).strip()
                    if not site or site.lower()=="nan": continue
                    records.append({'Country':df.at[idx,'Country'],'Site':site,'Raw':raw})
        site_stake_df = pd.DataFrame(records)
        def split_items(r):
            tmp = re.sub(r'\d+\.', ';', r)
            tmp = re.sub(r'\*+', ';', tmp)
            return [p.strip() for p in re.split(r'[;,\n]+', tmp) if p.strip()]
        clean = site_stake_df.assign(Stakeholder=site_stake_df['Raw'].apply(split_items)).explode('Stakeholder')
        def join_sites(s): return "; ".join(sorted({x for x in s if isinstance(x,str) and x.strip()}))
        grouped = clean.groupby(['Country','Stakeholder']).agg(
            SitesList=('Site', join_sites),
            CountSites=('Site', lambda s: s.nunique())
        ).reset_index()
        display_df = grouped.sort_values(['Country','CountSites','Stakeholder'],ascending=[True,False,True]).reset_index(drop=True)
        st.subheader("Stakeholders by Country")
        st.dataframe(display_df, use_container_width=True, height=400)
        st.download_button("Download Stakeholders (CSV)", display_df.to_csv(index=False),
                           "stakeholders.csv","text/csv")
        counts = clean.groupby('Stakeholder')['Site'].nunique().reset_index(name='CountSites').sort_values('CountSites',ascending=False)
        top5 = counts.head(5)
        st.subheader("Top 5 Stakeholders Across All Countries")
        st.table(top5.set_index('Stakeholder'))
        fig7 = px.bar(
            top5, x='Stakeholder', y='CountSites',
            title="Top 5 Most Common Stakeholders",
            color='Stakeholder', color_discrete_sequence=palette
        )
        fig7.update_layout(xaxis_title=None, yaxis_title="Number of Sites")
        st.plotly_chart(fig7, use_container_width=True)

    # ── Tab 8: Policy & Legislation ─────────────────────────────────────────────
    with tabs[7]:
        st.header("8. Policy & Legislation")
        col_exists = "Is there a health research policy in your country?"
        col_disseminated = "Has the policy been disseminated?"
        col_implemented = "Is the policy currently under implementation?"
        col_budget = ("What percentage of the national health budget is allocated "
                      "to health-related R&D, considering the AU's 2% target?")
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
        yes_set = {'yes','oui','checked'}
        to_bin = lambda x: 1 if str(x).strip().lower() in yes_set else 0

        df['PolicyExists']       = df[col_exists].map(to_bin)
        df['PolicyDisseminated'] = df[col_disseminated].map(to_bin)
        df['PolicyImplemented']  = df[col_implemented].map(to_bin)

        budget_series = df[col_budget].astype(str).str.rstrip('%').replace('', '0')
        df['Budget_pct'] = (
            pd.to_numeric(budget_series, errors="coerce")
              .fillna(0).clip(0,100) / 100.0
        )

        sop_flags = df[sop_cols].applymap(to_bin)
        df['SOP_Coverage'] = sop_flags.sum(axis=1) / len(sop_cols)

        site_policy = pd.DataFrame({
            'Country': df['Country'],
            'Exists': df['PolicyExists'],
            'Disseminated': df['PolicyDisseminated'] & df['PolicyExists'],
            'Implemented': df['PolicyImplemented'] & df['PolicyExists'],
            'Budget': df['Budget_pct'],
            'SOP_Coverage': df['SOP_Coverage']
        })

        country_summary = site_policy.groupby('Country').agg(
            pct_with_policy   = ('Exists','mean'),
            pct_disseminated  = ('Disseminated','mean'),
            pct_implemented   = ('Implemented','mean'),
            avg_budget_alloc  = ('Budget','mean'),
            avg_sop_coverage  = ('SOP_Coverage','mean'),
            num_sites         = ('Exists','count')
        ).reset_index()
        country_summary['implementation_gap'] = (
            country_summary['pct_with_policy'] - country_summary['pct_implemented']
        )

        disp = country_summary.copy()
        for p in ['pct_with_policy','pct_disseminated','pct_implemented','implementation_gap']:
            disp[p] = (disp[p]*100).round(1).astype(str) + '%'
        disp[['avg_budget_alloc','avg_sop_coverage']] = disp[['avg_budget_alloc','avg_sop_coverage']].round(2)
        st.subheader("Country-level Policy & Legislation Summary")
        st.table(disp.set_index('Country'))

        melt_bar = country_summary.melt(
            id_vars='Country',
            value_vars=['pct_with_policy','pct_disseminated','pct_implemented'],
            var_name='Metric', value_name='Value'
        )
        label_map = {
            'pct_with_policy': '% With Policy',
            'pct_disseminated': '% Disseminated',
            'pct_implemented':  '% Implemented'
        }
        melt_bar['Metric'] = melt_bar['Metric'].map(label_map)
        fig_bar = px.bar(
            melt_bar, x='Country', y='Value', color='Metric',
            barmode='group', color_discrete_sequence=palette,
            title="Policy Existence, Dissemination & Implementation by Country",
            labels={'Value':'Proportion (0–1)'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            melt_bar, names='Metric', values='Value', facet_col='Country',
            title="Policy Breakdown by Country (Proportions)",
            color_discrete_sequence=palette,
            labels={'Value':'Proportion (0–1)'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        melt_radar = country_summary.melt(
            id_vars='Country',
            value_vars=['pct_with_policy','pct_disseminated','pct_implemented','implementation_gap'],
            var_name='Metric', value_name='Value'
        )
        fig_radar = px.line_polar(
            melt_radar, r='Value', theta='Metric', color='Country',
            line_close=True, title='Policy Radar Chart by Country',
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.download_button(
            "Download Policy Summary (CSV)",
            country_summary.to_csv(index=False),
            "policy_summary.csv","text/csv"
        )

    progress.progress(100)

# ── Page Routing ───────────────────────────────────────────────────────────────
if st.session_state.page == "results":
    show_results()
else:
    show_upload()
