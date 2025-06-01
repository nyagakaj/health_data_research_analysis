import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import plotly.express as px
import country_converter as coco
import pycountry

# ────
# Page & Theme Setup 
# ────
st.set_page_config(
    page_title="Health Research Dashboard",
    layout="wide"
)

# Routing State 
if "page" not in st.session_state:
    st.session_state.page = "upload"

# ────
# Top Navigation Bar 
# ────
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


# Custom CSS 
st.markdown("""
<style>
  /* Tabs styling */
  .stTabs > div[role="tablist"] {
    display: flex !important;
    gap: 30px !important;            
    padding-bottom: 8px !important;  
  }
  .stTabs [role="tab"] {
    flex: 1 1 0 !important;           
    text-align: center !important;
    font-size: 18px !important;
    padding: 10px 22px !important;
    border-radius: 8px 8px 0 0 !important;
    margin: 0 !important;             
  }
  .stTabs [role="tab"]:nth-child(1) { background: #1A5632; }
  .stTabs [role="tab"]:nth-child(2) { background: #9F2241; }
  .stTabs [role="tab"]:nth-child(3) { background: #B4A269; }
  .stTabs [role="tab"]:nth-child(4) { background: #348F41; }
  .stTabs [role="tab"]:nth-child(5) { background: #58595B; }
  .stTabs [role="tab"]:nth-child(6) { background: #9F2241; }
  .stTabs [role="tab"]:nth-child(7) { background: #B4A269; }
  .stTabs [role="tab"]:nth-child(8) { background: #1A5632; }
  .stTabs [role="tab"]:nth-child(9) { background: #58595B; }
  .stTabs [role="tab"]:nth-child(10){ background: #348F41; }
  .stTabs [role="tab"][aria-selected="true"] {
    color: #fff !important;
  }

  /* Widen the Streamlit form container */
  .stForm {
    max-width: 1200px !important;
    width: 100% !important;
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

# Shared palette
palette = [
    "#1A5632", "#9F2241", "#B4A269", "#348F41",
    "#58595B", "#9F2241", "#B4A269", "#1A5632"
]

# Utility: ensure unique column names 
def make_unique(cols):
    cnt, out = {}, []
    for c in cols:
        cnt[c] = cnt.get(c, 0)
        out.append(c if cnt[c] == 0 else f"{c}_{cnt[c]}")
        cnt[c] += 1
    return out

# UPLOAD PAGE 
def show_upload():
    st.markdown('<div class="upload-form">', unsafe_allow_html=True)
    with st.form("upload_form"):
        st.markdown('<h2>Africa Research Sites Mapping Dashboard</h2>', unsafe_allow_html=True)
        st.markdown('<p class="caption">Upload one or both CSV files to get started.</p>', unsafe_allow_html=True)

        st.markdown('<div class="dataset-boxes">', unsafe_allow_html=True)
        # English dataset
        st.markdown('<div class="dataset-box"><h4>English Dataset</h4>', unsafe_allow_html=True)
        en_file = st.file_uploader("", type="csv", key="en_file")
        st.markdown('</div>', unsafe_allow_html=True)
        # French dataset
        st.markdown('<div class="dataset-box"><h4>French Dataset</h4>', unsafe_allow_html=True)
        fr_file = st.file_uploader("", type="csv", key="fr_file")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Store uploaded file bytes in session state
        if en_file is not None:
            st.session_state["en_bytes"] = en_file.getvalue()
        if fr_file is not None:
            st.session_state["fr_bytes"] = fr_file.getvalue()

        if st.form_submit_button("Analyze Data"):
            if ("en_bytes" not in st.session_state) and ("fr_bytes" not in st.session_state):
                st.error("Please upload at least one CSV file.")
            else:
                # Remove any previously cached DataFrame
                if "df_full" in st.session_state:
                    del st.session_state["df_full"]
                st.session_state.page = "results"
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# RESULTS PAGE 
def show_results():
    st.markdown("### Results")
    if st.button("← Back to Upload"):
        st.session_state.page = "upload"
        st.rerun()

    # Initialize progress bar
    progress = st.progress(0)

    # 1) Grab bytes from session_state
    en_bytes = st.session_state.get("en_bytes")
    fr_bytes = st.session_state.get("fr_bytes")

    # If neither uploader has bytes, show error and stop
    if (en_bytes is None) and (fr_bytes is None):
        st.error("No data to process. Please go back and upload at least one CSV.")
        return

    # Only load and cache the full DataFrame once
    if "df_full" not in st.session_state:
        # 2) Read whichever files were provided, from bytes
        df_en = pd.read_csv(io.BytesIO(en_bytes), keep_default_na=False) if en_bytes else pd.DataFrame()
        df_fr = pd.read_csv(io.BytesIO(fr_bytes), keep_default_na=False) if fr_bytes else pd.DataFrame()

        # 3) Header map (if both exist)
        if not df_en.empty and not df_fr.empty:
            df_fr.rename(columns=dict(zip(df_fr.columns, df_en.columns)), inplace=True)

        # 4) Harmonize Yes/No in French
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

        # 5) Align & concat
        all_cols = list(dict.fromkeys(df_en.columns.tolist() + df_fr.columns.tolist()))
        df_en = df_en.reindex(columns=all_cols, fill_value="")
        df_fr = df_fr.reindex(columns=all_cols, fill_value="")
        df = pd.concat([df_en, df_fr], ignore_index=True)
        blank_cols = [c for c in df.columns if (df[c] == "").all()]
        df.drop(columns=blank_cols, inplace=True)
        progress.progress(60)

        # 6) Filter & unify country
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

        # 7) Detect name column
        name_col = next(
            (c for c in df.columns if re.search(r'\bname\b', c, re.I)
                 or re.search(r'nom.*institut', c, re.I)),
            df.columns[0]
        )
        df.attrs["name_col"] = name_col  # store for later use
        progress.progress(80)

        # Cache the combined DataFrame
        st.session_state["df_full"] = df.copy()
    else:
        df = st.session_state["df_full"]
        name_col = df.attrs.get("name_col", df.columns[0])

    # 8) Deep-Dive selector
    st.subheader("Deep-Dive Configuration")
    countries = sorted(df["Country"].unique())
    selected_countries = st.multiselect(
        "Select one or more countries:",
        options=countries
    )
    df_deep = df[df["Country"].isin(selected_countries)].copy() if selected_countries else pd.DataFrame()

    # 9) Precompute shared flags/tables 
    # 9.1 Identification categories
    cats = {
        "BasicScience":[r"\bbasic\b",r"fundamental"],
        "Preclinical":[r"preclinical"],
        "ClinicalTrials":[r"clinical"],
        "Epidemiological":[r"epidemiolog"]
    }
    # 9.2 Human resource boolean & numeric groups
    bool_groups = {
        "Clinical Staff":[r"availability of clinical staff"],
        "Lab Staff":[r"availability of laboratory staff"],
        "Pharmacy Staff":[r"availability of pharmacy staff"],
        "Bioinformatics":[r"bioinformatics"],
        "Cell Culture":[r"cell culture"],
        "Org. Synthesis":[r"organic synthesis"],
        "Virology":[r"virology"],
    }
    num_groups = {
        "Other Staff":[r"number of other staff"],
        "PhD":[r"doctorate|phd"],
        "MSc":[r"master's|msc"],
    }
    # 9.3 Stakeholder explosion
    free_cols = [
        'Other (Please specify)',
        'If yes, list the research collaborations in the last 5 years'
    ]
    records = []
    for col in free_cols:
        if col in df.columns:
            exploded = df[col].astype(str).str.split(r'[;,]+').explode().str.strip().dropna()
            for idx, raw in exploded.items():
                if not raw or raw.lower() == "nan":
                    continue
                site = str(df.at[idx, name_col]).strip()
                if not site or site.lower() == "nan":
                    continue
                records.append({'Country': df.at[idx, 'Country'], 'Site': site, 'RawEntry': raw})
    site_stake_df = pd.DataFrame(records)

    def split_items(r):
        tmp = re.sub(r'\d+\.', ';', r)
        tmp = re.sub(r'\*+', ';', tmp)
        return [p.strip() for p in re.split(r'[;,\n]+', tmp) if p.strip()]

    site_clean = site_stake_df.assign(Stakeholder=site_stake_df['RawEntry'].apply(split_items)).explode('Stakeholder')

    # 9.4 Policy flags
    policy_exists_col       = "Is there a health research policy in your country?"
    policy_disseminated_col = "Has the policy been disseminated?"
    policy_implemented_col  = "Is the policy currently under implementation?"
    budget_col = ("What percentage of the national health budget is allocated "
                  "to health-related R&D, considering the AU's 2% target?")
    sop_cols = [c for c in df.columns if c.startswith('Available SOPs')]

    def to_bin(x):
        return 1 if str(x).strip().lower() in ('yes','oui','checked') else 0

    df['PolicyExists']       = df[policy_exists_col].map(to_bin)
    df['PolicyDisseminated'] = df[policy_disseminated_col].map(to_bin)
    df['PolicyImplemented']  = df[policy_implemented_col].map(to_bin)
    df['Budget_pct'] = (
        pd.to_numeric(df[budget_col].astype(str).str.rstrip('%').replace('', '0'),
                      errors='coerce')
          .fillna(0).clip(0,100) / 100.0
    )
    df['SOP_Coverage'] = df[sop_cols].applymap(to_bin).sum(axis=1) / len(sop_cols)

    site_policy = pd.DataFrame({
        'Country':      df['Country'],
        'Exists':       df['PolicyExists'],
        'Disseminated': df['PolicyDisseminated'] & df['PolicyExists'],
        'Implemented':  df['PolicyImplemented'] & df['PolicyExists'],
        'Budget':       df['Budget_pct'],
        'SOP_Coverage': df['SOP_Coverage']
    })

    # 9.5 Build HR columns in full df
    for name, pats in bool_groups.items():
        cols = [c for c in df.columns if any(re.search(p, c, re.I) for p in pats)]
        df[name] = df[cols].eq("Yes").any(axis=1).astype(int) if cols else 0

    for name, pats in num_groups.items():
        cols = [c for c in df.columns if any(re.search(p, c, re.I) for p in pats)]
        if cols:
            df[name] = df[cols].apply(pd.to_numeric, errors='coerce').max(axis=1).fillna(0).astype(int)
        else:
            df[name] = 0

    # 10) Compute Maps DataFrame 
    # 10.1 Identification flags
    for cat, pats in cats.items():
        cols = [c for c in df.columns if any(re.search(p, c, re.I) for p in pats)]
        df[f"Is{cat}"] = df[cols].eq("Yes").any(axis=1)
    # 10.2 CapabilityScore
    df["CapabilityScore"] = df[[f"Is{cat}" for cat in cats]].sum(axis=1)
    cap_df = df.groupby("Country")["CapabilityScore"].mean().reset_index(name="Avg Capability")
    # 10.3 Phase I
    trans_cols = [c for c in df.columns if re.search(r"phase.*i", c, re.I)]
    df["HasPhaseI"] = df[trans_cols].eq("Yes").any(axis=1)
    tr_df = df.groupby("Country")["HasPhaseI"].sum().reset_index(name="Phase I Sites")
    # 10.4 InfraIndex
    infra_terms = ["availability of advanced","level of biosecurity","iso certification"]
    infra_cols = [c for c in df.columns if any(t in c.lower() for t in infra_terms)]
    df["InfraIndex"] = df[infra_cols].eq("Yes").sum(axis=1)
    infra_df = df.groupby("Country")["InfraIndex"].mean().reset_index(name="Avg InfraIndex")
    # 10.5 IRB
    ethic_terms = ["ethic","irb","regul","guidelines"]
    ethic_cols = [c for c in df.columns if any(t in c.lower() for t in ethic_terms)]
    df["HasIRB"] = df[ethic_cols].eq("Yes").any(axis=1)
    er_df = df.groupby("Country")["HasIRB"].sum().reset_index(name="IRB Sites")
    # 10.6 Policy %
    pol_df = site_policy.groupby("Country")["Exists"].mean().reset_index(name="% With Policy")

    # Merge into map_df
    map_df = (
        cap_df
        .merge(tr_df, on="Country")
        .merge(infra_df, on="Country")
        .merge(er_df, on="Country")
        .merge(pol_df, on="Country")
    )
    map_df["Country"] = map_df["Country"].str.strip()
    cc = coco.CountryConverter()
    map_df["ISO_A3"] = cc.convert(names=map_df["Country"], to="ISO3", not_found=None)

    def fuzzy_iso(name):
        try:
            return pycountry.countries.get(name=name).alpha_3
        except:
            try:
                return pycountry.countries.search_fuzzy(name)[0].alpha_3
            except:
                return None

    missing_iso = map_df["ISO_A3"].isnull()
    if missing_iso.any():
        map_df.loc[missing_iso, "ISO_A3"] = map_df.loc[missing_iso, "Country"].apply(fuzzy_iso)
        still_missing = map_df.loc[map_df["ISO_A3"].isnull(), "Country"].unique()
        if len(still_missing):
            st.warning("Couldn't map to ISO3: " + ", ".join(still_missing))

    map_long = map_df.melt(id_vars=["Country","ISO_A3"], var_name="Metric", value_name="Value")
    progress.progress(95)

    
    # Initialize Tabs  
    tabs = st.tabs([
        "1. Identification","2. Capacity","3. Human Resources",
        "4. Translational","5. Infrastructure","6. Ethics/Reg",
        "7. Stakeholders","8. Policy","9. Deep-Dive","10. Maps"
    ])

    # Tab 1: Identification 
    with tabs[0]:
        st.header("1. Identification of Research Sites")
        df_current = df if not selected_countries else df_deep
        bool_cols = []
        for cat, pats in cats.items():
            cols = [c for c in df_current.columns if any(re.search(p, c, re.I) for p in pats)]
            df_current[f"Is{cat}"] = df_current[cols].eq("Yes").any(axis=1)
            bool_cols.append(f"Is{cat}")

        summary1 = df_current.groupby('Country')[bool_cols].sum().rename(columns=lambda x: x.replace("Is",""))
        other_mask = ~df_current[bool_cols].any(axis=1)
        summary1["Other"] = df_current[other_mask].groupby("Country").size().reindex(summary1.index, fill_value=0)
        st.table(summary1)

        melt1 = summary1.reset_index().melt('Country', var_name='Category', value_name='Count')
        fig1 = px.bar(
            melt1, x='Country', y='Count', color='Category',
            barmode='group', title="Sites by Category & Country",
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig1b = px.sunburst(
            melt1, path=['Country','Category'], values='Count',
            title='Sunburst of Research Sites by Country and Category',
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig1b, use_container_width=True)

    # Tab 2: Capacity 
    with tabs[1]:
        st.header("2. Capacity Evaluation")
        fig2 = px.bar(
            cap_df, x='Country', y='Avg Capability', color='Country',
            title="Avg Capability Score by Country", color_discrete_sequence=palette
        )
        st.plotly_chart(fig2, use_container_width=True)

        heat = melt1.pivot(index='Country', columns='Category', values='Count').fillna(0)
        fig2b = px.imshow(
            heat, labels=dict(x="Category", y="Country", color="Count"),
            title="Heatmap of Site Counts per Category & Country",
            color_continuous_scale=["#D0E8D8","#1A5632"], zmin=0, zmax=heat.values.max()
        )
        fig2b.update_layout(height=500, margin=dict(t=50,b=50))
        st.plotly_chart(fig2b, use_container_width=True)

    # Tab 3: Human Resources 
    with tabs[2]:
        st.header("3. Human Resource Assessment")
        bool_sum = df.groupby('Country')[list(bool_groups.keys())].sum()
        num_sum = df.groupby('Country')[list(num_groups.keys())].sum()
        total = bool_sum.add(num_sum, fill_value=0).sum(axis=1).astype(int)
        num_sum["Total Staff"] = total
        combined = pd.concat([bool_sum, num_sum], axis=1)
        combined.index.name = "Country"
        st.table(combined)

        melt_bool = bool_sum.reset_index().melt('Country', var_name='Indicator', value_name='Count of Yes')
        fig_bool = px.bar(
            melt_bool, x="Country", y="Count of Yes", color="Indicator",
            barmode="group", title="Sites Reporting “Yes” by Indicator",
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig_bool, use_container_width=True)

        melt_num = num_sum.reset_index().melt('Country', var_name='Staff Category', value_name='Count')
        fig_num = px.bar(
            melt_num, x="Country", y="Count", color="Staff Category", barmode="group",
            title="Staff Counts by Country (incl. Total Staff)", color_discrete_sequence=palette
        )
        st.plotly_chart(fig_num, use_container_width=True)

    # Tab 4: Translational 
    with tabs[3]:
        st.header("4. Translational Research (Phase I)")
        fig4 = px.bar(
            tr_df, x='Country', y='Phase I Sites', color='Country',
            title="Sites Reporting Phase I Trials", color_discrete_sequence=palette,
            range_y=[0, tr_df['Phase I Sites'].max()+1]
        )
        st.plotly_chart(fig4, use_container_width=True)

        cap_df_renamed = cap_df.rename(columns={"Avg Capability":"CapabilityScore"})
        cap_tr = cap_df_renamed.merge(tr_df, on='Country')
        fig4b = px.scatter(
            cap_tr, x='CapabilityScore', y='Phase I Sites', size='Phase I Sites',
            color='Country', title='Phase I Trials vs. Capability Score',
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig4b, use_container_width=True)

    # Tab 5: Infrastructure 
    with tabs[4]:
        st.header("5. Infrastructure Analysis")
        fig5 = px.bar(
            infra_df, x='Country', y='Avg InfraIndex', color='Country',
            title="Avg Infrastructure Index by Country", color_discrete_sequence=palette,
            range_y=[0, infra_df['Avg InfraIndex'].max()+1]
        )
        st.plotly_chart(fig5, use_container_width=True)

        violin_df = df[['Country','InfraIndex']].copy()
        fig5b = px.violin(
            violin_df, x='Country', y='InfraIndex',
            title="Infrastructure Index Distribution by Country",
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig5b, use_container_width=True)

    # Tab 6: Ethics & Regulatory 
    with tabs[5]:
        st.header("6. Ethics & Regulatory")
        fig6 = px.bar(
            er_df, x='Country', y='IRB Sites', color='Country',
            title="Sites with In‐house IRBs by Country", color_discrete_sequence=palette
        )
        st.plotly_chart(fig6, use_container_width=True)

        pie_df = df.groupby(['Country','HasIRB']).size().reset_index(name='Count')
        fig6b = px.pie(
            pie_df, names='HasIRB', values='Count', facet_col='Country',
            title='IRB Coverage by Country', color_discrete_sequence=palette
        )
        st.plotly_chart(fig6b, use_container_width=True)

    # Tab 7: Stakeholder Mapping  
    with tabs[6]:
        st.header("7. Stakeholder Mapping")
        grouped_full = (
            site_clean
            .groupby(['Country','Stakeholder'])
            .agg(
                SitesList=('Site', lambda s: "; ".join(sorted(set(s)))),
                CountSites=('Site', lambda s: s.nunique())
            )
            .reset_index()
            .sort_values(['Country','CountSites'], ascending=[True,False])
        )
        st.dataframe(grouped_full, use_container_width=True, height=400)
        st.download_button("Download Stakeholders (CSV)", grouped_full.to_csv(index=False),
                           "stakeholders.csv","text/csv")
        stake_counts = (
            site_clean
            .groupby('Stakeholder')['Site']
            .nunique()
            .reset_index(name='CountSites')
            .sort_values('CountSites', ascending=False)
        )
        top5 = stake_counts.head(5)
        st.subheader("Top 5 Stakeholders (All Countries)")
        st.table(top5.set_index('Stakeholder'))
        fig_top5 = px.bar(
            top5, x='Stakeholder', y='CountSites',
            color='Stakeholder', title="Top 5 Stakeholders",
            color_discrete_sequence=palette
        )
        st.plotly_chart(fig_top5, use_container_width=True)

    # Tab 8: Policy & Legislation 
    with tabs[7]:
        st.header("8. Policy & Legislation")
        country_summary = site_policy.groupby('Country').agg(
            pct_with_policy   = ('Exists','mean'),
            pct_disseminated  = ('Disseminated','mean'),
            pct_implemented   = ('Implemented','mean'),
            avg_budget_alloc  = ('Budget','mean'),
            avg_sop_coverage  = ('SOP_Coverage','mean'),
            num_sites         = ('Exists','count')
        ).reset_index()
        country_summary['implementation_gap'] = country_summary['pct_with_policy'] - country_summary['pct_implemented']
        disp = country_summary.copy()
        for p in ['pct_with_policy','pct_disseminated','pct_implemented','implementation_gap']:
            disp[p] = (disp[p]*100).round(1).astype(str) + '%'
        disp[['avg_budget_alloc','avg_sop_coverage']] = disp[['avg_budget_alloc','avg_sop_coverage']].round(2)
        st.table(disp.set_index('Country'))

        melt_bar = country_summary.melt(
            id_vars='Country',
            value_vars=['pct_with_policy','pct_disseminated','pct_implemented'],
            var_name='Metric', value_name='Value'
        )
        label_map = {
            'pct_with_policy':'% With Policy',
            'pct_disseminated':'% Disseminated',
            'pct_implemented':'% Implemented'
        }
        melt_bar['Metric'] = melt_bar['Metric'].map(label_map)
        fig_bar = px.bar(
            melt_bar, x='Country', y='Value', color='Metric', barmode='group',
            color_discrete_sequence=palette, title="Policy Metrics by Country"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            melt_bar, names='Metric', values='Value', facet_col='Country',
            title="Policy Breakdown by Country", color_discrete_sequence=palette,
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

    # Tab 9: Deep-Dive 
    with tabs[8]:
        st.header("9. Deep-Dive")
        if not selected_countries:
            st.info("Select one or more countries above to see details.")
        elif df_deep.empty:
            st.warning("No records found for the selected country(ies).")
        elif len(selected_countries) == 1:
            country = selected_countries[0]
            st.subheader(f"Deep-Dive: {country}")

            # Identification (visual)
            st.markdown("**Identification**")
            summary_id_single = (
                df_deep.groupby('Country')[[f"Is{c}" for c in cats]]
                       .sum()
                       .rename(columns=lambda x: x.replace("Is",""))
            )
            fig_id = px.bar(
                summary_id_single.reset_index().melt('Country', var_name='Category', value_name='Count'),
                x='Category', y='Count', title="Identification Breakdown"
            )
            st.plotly_chart(fig_id, use_container_width=True)

            # Capacity (visual)
            st.markdown("**Capacity Score**")
            cap_single = df_deep[[f"Is{c}" for c in cats]].sum(axis=1).mean().round(2) if not df_deep.empty else 0
            st.metric("Avg Capability", cap_single)

            # Human Resources (visual)
            st.markdown("**Human Resources**")
            bool_sum_single = df_deep[list(bool_groups.keys())].sum() if all(col in df_deep.columns for col in bool_groups.keys()) else pd.Series(0, index=list(bool_groups.keys()))
            num_sum_single = df_deep[list(num_groups.keys())].sum() if all(col in df_deep.columns for col in num_groups.keys()) else pd.Series(0, index=list(num_groups.keys()))
            hr_plot_df = pd.DataFrame({
                'Indicator': bool_sum_single.index.tolist(),
                'Count': bool_sum_single.values
            })
            fig_hr = px.bar(hr_plot_df, x='Indicator', y='Count', title="Human Resource “Yes” Counts")
            st.plotly_chart(fig_hr, use_container_width=True)

            # Phase I (visual)
            st.markdown("**Phase I Trials**")
            phase_val = int(df_deep['HasPhaseI'].sum()) if 'HasPhaseI' in df_deep.columns else 0
            st.metric("Phase I Sites", phase_val)

            # Infrastructure (visual)
            st.markdown("**Infrastructure Index**")
            infra_vals = df_deep['InfraIndex'] if 'InfraIndex' in df_deep.columns else pd.Series(dtype=float)
            st.bar_chart(infra_vals)

            # Ethics/Regulatory (visual)
            st.markdown("**Ethics & Regulatory**")
            irb_val = int(df_deep['HasIRB'].sum()) if 'HasIRB' in df_deep.columns else 0
            st.metric("In-house IRB Sites", irb_val)

            # Stakeholders (table only)
            st.markdown("**Key Stakeholders**")
            stakeholders_single = (
                site_clean[site_clean['Country'] == country]
                .groupby('Stakeholder')
                .agg(
                    SitesList=('Site', lambda s: "; ".join(sorted(set(s)))),
                    CountSites=('Site', lambda s: s.nunique())
                )
                .reset_index()
                .sort_values('CountSites', ascending=False)
            )
            st.table(stakeholders_single[['Stakeholder','CountSites','SitesList']])

            # Policy & Legislation (visual)
            st.markdown("**Policy & Legislation**")
            exists_count = int(site_policy[site_policy['Country']==country]['Exists'].sum()) if 'Exists' in site_policy.columns else 0
            avg_budget = site_policy[site_policy['Country']==country]['Budget'].mean()*100 if 'Budget' in site_policy.columns else 0
            avg_sop = site_policy[site_policy['Country']==country]['SOP_Coverage'].mean()*100 if 'SOP_Coverage' in site_policy.columns else 0
            st.metric("Policy Exists (count)", exists_count)
            st.metric("Avg Budget (%)", f"{avg_budget:.1f}%")
            st.metric("Avg SOP Coverage (%)", f"{avg_sop:.1f}%")

        else:
            st.subheader(f"Deep-Dive Comparison: {', '.join(selected_countries)}")

            # Identification Comparison
            st.markdown("**Identification Comparison**")
            summary_id_multi = df_deep.groupby('Country')[[f"Is{c}" for c in cats]].sum()\
                                     .rename(columns=lambda x: x.replace("Is",""))
            st.dataframe(summary_id_multi.loc[selected_countries])

            # Capacity Comparison
            st.markdown("**Capacity Score Comparison**")
            cap_multi = df_deep.groupby('Country')[[f"Is{c}" for c in cats]].sum().mean(axis=1).round(2)
            st.bar_chart(cap_multi)

            # Human Resources Comparison
            st.markdown("**Human Resources Comparison**")
            bool_sum_multi = df_deep.groupby('Country')[list(bool_groups.keys())].sum()
            st.dataframe(bool_sum_multi.loc[selected_countries])

            # Phase I Comparison
            st.markdown("**Phase I Trials Comparison**")
            phase_multi = tr_df.set_index('Country').loc[selected_countries]
            st.dataframe(phase_multi)

            # Infrastructure Comparison
            st.markdown("**Infrastructure Index Comparison**")
            infra_multi = infra_df.set_index('Country').loc[selected_countries]
            st.dataframe(infra_multi)

            # Ethics/Regulatory Comparison
            st.markdown("**Ethics & Regulatory Comparison**")
            er_multi = er_df.set_index('Country').loc[selected_countries]
            st.dataframe(er_multi)

            # Stakeholders Comparison (table only)
            st.markdown("**Key Stakeholders Comparison**")
            stakeholders_multi = (
                site_clean[site_clean['Country'].isin(selected_countries)]
                .groupby(['Country','Stakeholder'])
                .agg(
                    SitesList=('Site', lambda s: "; ".join(sorted(set(s)))),
                    CountSites=('Site', lambda s: s.nunique())
                )
                .reset_index()
                .sort_values(['Country','CountSites'], ascending=[True,False])
            )
            st.dataframe(
                stakeholders_multi[['Country','Stakeholder','CountSites','SitesList']],
                use_container_width=True
            )

            # Policy Comparison
            st.markdown("**Policy & Legislation Comparison**")
            policy_multi = site_policy.groupby('Country').agg(
                Exists_Count=('Exists','sum'),
                Avg_Budget_pct=('Budget','mean'),
                Avg_SOP_Coverage=('SOP_Coverage','mean')
            ).loc[selected_countries].reset_index()
            policy_multi['Avg_Budget_pct'] = policy_multi['Avg_Budget_pct'] * 100
            policy_multi['Avg_SOP_Coverage'] = policy_multi['Avg_SOP_Coverage'] * 100
            policy_multi = policy_multi.rename(columns={
                'Exists_Count': 'Exists (count)',
                'Avg_Budget_pct': 'Avg Budget (%)',
                'Avg_SOP_Coverage': 'Avg SOP Coverage (%)'
            })
            st.dataframe(policy_multi.set_index('Country'))

    # Tab 10: Maps 
    with tabs[9]:
        st.header("10. Spatial Overview of Core Metrics")
        metrics = map_long["Metric"].unique()
        for metric in metrics:
            st.subheader(metric)
            df_m = map_long[map_long["Metric"] == metric]
            fig = px.choropleth(
                df_m,
                locations="ISO_A3",
                locationmode="ISO-3",
                hover_name="Country",
                color="Value",
                scope="africa",
                color_continuous_scale=["#D0E8D8","#1A5632"],
                title=metric
            )
            fig.update_geos(
                visible=False,
                showland=True,
                landcolor="lightgray",
                showcountries=True,
                countrycolor="white"
            )
            fig.update_layout(
                margin=dict(t=50, b=0, l=0, r=0),
                height=800
            )
            st.plotly_chart(fig, use_container_width=True)

            html = fig.to_html()
            st.download_button(
                label=f"Download {metric} map (HTML)",
                data=html,
                file_name=f"{metric.replace(' ', '_').lower()}_map.html",
                mime="text/html"
            )

    progress.progress(100)

# Page Routing 
if st.session_state.page == "results":
    show_results()
else:
    show_upload()
