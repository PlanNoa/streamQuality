import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Level Product Analytics",
    layout="wide",
)

# read csv from a URL
@st.cache_data
def get_data(path):
    df = pd.read_excel(path[0], sheet_name="Geolocation Accuracy", header=[0, 1], index_col=[0])
    df = df.fillna("NA")
    df = df[df["SW ID"]["EOLP"] != "NA"]
    for i in range(1, len(path)):
        df_temp = pd.read_excel(path[i], sheet_name="Geolocation Accuracy", header=[0, 1], index_col=[0])
        df_temp = df_temp.fillna("NA")
        df_temp = df_temp[df_temp["SW ID"]["EOLP"] != "NA"]
        df = pd.concat([df, df_temp])
    df = df.replace("NA", np.nan)
    return df

# dashboard title
st.title("Level Product Analytics")

# 폴더/파일 경로 검색
file = st.file_uploader('File Uploader:', type="xlsx", accept_multiple_files=True)
if len(file) > 0:
    df = get_data(file)

    # filters
    with st.expander("Search Filter"):
        EOLP = st.multiselect("SW ID / EOLP",
                              df["SW ID"]["EOLP"].dropna().unique(),
                              default=df["SW ID"]["EOLP"].dropna().unique())
        FSW = st.multiselect("SW ID / FSW",
                             df["SW ID"]["FSW"].dropna().unique(),
                             default=df["SW ID"]["FSW"].dropna().unique())
        start_date_col, end_date_col = st.columns(2)
        with start_date_col:
            start_date = st.date_input("Search Date Range Start",
                                       np.nanmin(df["Satellite Status"]["Strip Imaging Start Time"].dt.date))
        with end_date_col:
            end_date = st.date_input("Search Date Range End",
                                     np.nanmax(df["Satellite Status"]["Strip Imaging Start Time"].dt.date))
        Duration = st.slider("Sunlight Duration",
                             float(np.nanmin(df["Sunlit"]["Duration\n(sec)"]) - 1e-10),
                             float(np.nanmax(df["Sunlit"]["Duration\n(sec)"]) + 1e-10),
                             (float(np.nanmin(df["Sunlit"]["Duration\n(sec)"])),
                              float(np.nanmax(df["Sunlit"]["Duration\n(sec)"]))))
        STSFlag = st.multiselect("STS / STSFlag", pd.unique(df["STS"]["STSFlag"]),
                                 default=pd.unique(df["STS"]["STSFlag"]))
        Mode = st.multiselect("Operation / Mode", pd.unique(df["Operation"]["Mode"]),
                              default=pd.unique(df["Operation"]["Mode"]))
        lat_col, lon_col = st.columns(2)
        with lat_col:
            lat = st.slider("Scene Center Lat",
                            -90., 90.,
                            (-90., 90.))
        with lon_col:
            lon = st.slider("Scene Center Lon",
                            -180., 180.,
                            (-180., 180.))
        roll_col, pitch_col, yaw_col = st.columns(3)
        with roll_col:
            roll = st.slider("Attitude Roll",
                             -90., 90.,
                             (float(np.nanmin(df["Attitude"]["Roll\n(deg)"])),
                              float(np.nanmax(df["Attitude"]["Roll\n(deg)"]))))
        with pitch_col:
            pitch = st.slider("Attitude Pitch",
                              -90., 90.,
                              (float(np.nanmin(df["Attitude"]["Pitch\n(deg)"])),
                               float(np.nanmax(df["Attitude"]["Pitch\n(deg)"]))))
        with yaw_col:
            yaw = st.slider("Attitude Yaw",
                            -180., 180.,
                            (float(np.nanmin(df["Attitude"]["Yaw\n(deg)"])),
                             float(np.nanmax(df["Attitude"]["Yaw\n(deg)"]))))
        Height_min_col, Height_max_col, Height_maxdiff_col = st.columns(3)
        with Height_min_col:
            Height_min = st.slider("Height Min",
                                   float(np.nanmin(df["Height"]["Min.\n(m)"]) - 1e-10),
                                   float(np.nanmax(df["Height"]["Min.\n(m)"]) + 1e-10),
                                   (float(np.nanmin(df["Height"]["Min.\n(m)"])),
                                    float(np.nanmax(df["Height"]["Min.\n(m)"]))))
        with Height_max_col:
            Height_max = st.slider("Height Max",
                                   float(np.nanmin(df["Height"]["Max.\n(m)"]) - 1e-10),
                                   float(np.nanmax(df["Height"]["Max.\n(m)"]) + 1e-10),
                                   (float(np.nanmin(df["Height"]["Max.\n(m)"])),
                                    float(np.nanmax(df["Height"]["Max.\n(m)"]))))
        with Height_maxdiff_col:
            Height_maxdiff = st.slider("Height MaxDiff",
                                       float(np.nanmin(df["Height"]["MaxDiff.\n(m)"]) - 1e-10),
                                       float(np.nanmax(df["Height"]["MaxDiff.\n(m)"]) + 1e-10),
                                       (float(np.nanmin(df["Height"]["MaxDiff.\n(m)"])),
                                        float(np.nanmax(df["Height"]["MaxDiff.\n(m)"]))))
        CloudRatio = st.slider("Cloud Ratio",
                               0, 100,
                               (0, 100))

    # creating a single-element container
    placeholder = st.empty()

    # near real-time / live feed simulation
    while True:
        with placeholder.container():
            if file is not None:
                df = get_data(file)  # "data/R-Quality_02.ImageProduct_20220912_080004.xlsx"

                # filter dataframe
                df = df[(df["SW ID"]["EOLP"].isin(EOLP))]
                df = df[(df["SW ID"]["FSW"].isin(FSW))]
                df = df[(start_date <= df["Satellite Status"]["Strip Imaging Start Time"].dt.date) & (
                        df["Satellite Status"]["Strip Imaging Start Time"].dt.date <= end_date)]
                df = df[
                    (Duration[0] <= df["Sunlit"]["Duration\n(sec)"]) & (df["Sunlit"]["Duration\n(sec)"] <= Duration[1])]
                df = df[(df["STS"]["STSFlag"].isin(STSFlag))]
                df = df[(df["Operation"]["Mode"].isin(Mode))]
                df = df[(lat[0] <= df["Scene Center"]["Lat(deg)"]) & (df["Scene Center"]["Lat(deg)"] <= lat[1])]
                df = df[(lon[0] <= df["Scene Center"]["Lon(deg)"]) & (df["Scene Center"]["Lon(deg)"] <= lon[1])]
                df = df[(roll[0] <= df["Attitude"]["Roll\n(deg)"]) & (df["Attitude"]["Roll\n(deg)"] <= roll[1])]
                df = df[(pitch[0] <= df["Attitude"]["Pitch\n(deg)"]) & (df["Attitude"]["Pitch\n(deg)"] <= pitch[1])]
                df = df[(yaw[0] <= df["Attitude"]["Yaw\n(deg)"]) & (df["Attitude"]["Yaw\n(deg)"] <= yaw[1])]
                df = df[(Height_min[0] <= df["Height"]["Min.\n(m)"]) & (df["Height"]["Min.\n(m)"] <= Height_min[1])]
                df = df[(Height_max[0] <= df["Height"]["Max.\n(m)"]) & (df["Height"]["Max.\n(m)"] <= Height_max[1])]
                df = df[
                    (Height_maxdiff[0] <= df["Height"]["MaxDiff.\n(m)"]) & (
                                df["Height"]["MaxDiff.\n(m)"] <= Height_maxdiff[1])]
                df = df[(CloudRatio[0] <= df["Cloud"]["Ratio(%)"]) & (df["Cloud"]["Ratio(%)"] <= CloudRatio[1])]

                plt.close('all')
                # create two columns for charts
                df_len = len(df)
                max_ce90 = int(np.nanmax(df["Geolocation Error\n(Without GCP)"]["CE90\n(m)"]))
                max_avg = int(np.nanmax(df["Geolocation Error\n(Without GCP)"]["Average\n(m)"]))
                min_across = int(np.nanmin(df["Geolocation Error\n(Without GCP)"]["Average\n(Across)\n(m)(ColIdx:+)"]))
                max_across = int(np.nanmax(df["Geolocation Error\n(Without GCP)"]["Average\n(Across)\n(m)(ColIdx:+)"]))
                mod_across = max_across % 100
                min_along = int(np.nanmin(df["Geolocation Error\n(Without GCP)"]["Average\n(Along)\n(m)(LineNo:+)"]))
                max_along = int(np.nanmax(df["Geolocation Error\n(Without GCP)"]["Average\n(Along)\n(m)(LineNo:+)"]))
                mod_along = max_along % 100
                min_roll = int(np.nanmin(np.array(df["Attitude"]["Roll\n(deg)"])))
                max_roll = int(np.nanmax(np.array(df["Attitude"]["Roll\n(deg)"])))
                mod_roll = max_roll % 20
                min_pitch = int(np.nanmin(np.array(df["Attitude"]["Pitch\n(deg)"])))
                max_pitch = int(np.nanmax(np.array(df["Attitude"]["Pitch\n(deg)"])))
                mod_pitch = max_pitch % 20

                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    fig, ax = plt.subplots(figsize=(13, 5))
                    ax.plot(np.array(df["Geolocation Error\n(Without GCP)"]["CE90\n(m)"]), label="CE90", marker="o")
                    ax.set_xticks(range(0, df_len, 500))
                    ax.set_yticks(range(0, max_ce90, 500))
                    ax.grid()
                    ax.set_xlabel("image number")
                    ax.set_ylabel("CE90 (m)")
                    ax.legend()
                    ax.set_title("Geolocation Error (Without GCP)\nCE90 (m)")
                    st.pyplot(fig)

                with fig_col2:
                    fig, ax = plt.subplots(figsize=(13, 5))
                    ax.plot(np.array(df["Geolocation Error\n(Without GCP)"]["Average\n(m)"]), label="Avg.", marker="o")
                    ax.set_xticks(range(0, df_len, 500))
                    ax.set_yticks(range(0, max_avg, 100))
                    ax.grid()
                    ax.set_xlabel("image number")
                    ax.set_ylabel("Average (m)")
                    ax.legend()
                    ax.set_title("Geolocation Error (Without GCP)\nAverage (m)")
                    st.pyplot(fig)

                fig_col3, fig_col4 = st.columns(2)

                with fig_col3:
                    fig, ax = plt.subplots(figsize=(13, 5))
                    ax.plot(np.array(df["Geolocation Error\n(Without GCP)"]["Average\n(Across)\n(m)(ColIdx:+)"]), label="Avg. Across", marker="o")
                    ax.set_xticks(range(0, df_len, 500))
                    ax.set_yticks(range(max_across - mod_across + 100, min_across - mod_across, -100))

                    ax.grid()
                    ax.set_xlabel("image number")
                    ax.set_ylabel("Avg. Across Error (m)")
                    ax.legend()
                    ax.set_title("Geolocation Error (Without GCP)\nAverage Across Error (m)")
                    st.pyplot(fig)

                with fig_col4:
                    fig, ax = plt.subplots(figsize=(13, 5))
                    ax.plot(np.array(df["Geolocation Error\n(Without GCP)"]["Average\n(Along)\n(m)(LineNo:+)"]), label="Avg. Along", marker="o")
                    ax.set_xticks(range(0, df_len, 500))
                    ax.set_yticks(range(max_along - mod_along + 100, min_along - mod_along, -100))
                    ax.grid()
                    ax.set_xlabel("image number")
                    ax.set_ylabel("Avg. Along Error (m)")
                    ax.legend()
                    ax.set_title("Geolocation Error (Without GCP)\nAverage Along Error (m)")
                    st.pyplot(fig)

                fig_col5, fig_col6 = st.columns(2)
                with fig_col5:
                    fig, ax = plt.subplots(figsize=(13, 5))
                    ax.plot(np.array(df["Attitude"]["Roll\n(deg)"]), np.array(df["Geolocation Error\n(Without GCP)"]["Average\n(Across)\n(m)(ColIdx:+)"]), label="Avg. Across", marker="o", linestyle='None')
                    ax.set_xticks(range(max_roll - mod_roll + 20, min_roll - mod_roll, -20))
                    ax.set_yticks(range(max_across - mod_across + 100, min_across - mod_across, -100))
                    ax.grid()
                    ax.set_xlabel("Roll tilt (deg)")
                    ax.set_ylabel("Avg. Across Error (m)")
                    ax.legend()
                    ax.set_title("Geolocation Error (Without GCP)\nRoll vs Average Across Error (m)")
                    st.pyplot(fig)

                with fig_col6:
                    fig, ax = plt.subplots(figsize=(13, 5))
                    ax.plot(np.array(df["Attitude"]["Roll\n(deg)"]), np.array(df["Geolocation Error\n(Without GCP)"]["Average\n(Along)\n(m)(LineNo:+)"]), label="Avg. Along", marker="o", linestyle='None')
                    ax.set_xticks(range(max_roll - mod_roll + 20, min_roll - mod_roll, -20))
                    ax.set_yticks(range(max_along - mod_along + 100, min_along - mod_along, -100))
                    ax.grid()
                    ax.set_xlabel("Roll tilt (deg)")
                    ax.set_ylabel("Avg. Along Error (m)")
                    ax.legend()
                    ax.set_title("Geolocation Error (Without GCP)\nRoll vs Average Along Error (m)")
                    st.pyplot(fig)

                fig_col7, fig_col8 = st.columns(2)
                with fig_col7:
                    fig, ax = plt.subplots(figsize=(13, 5))
                    ax.plot(np.array(df["Attitude"]["Pitch\n(deg)"]), np.array(df["Geolocation Error\n(Without GCP)"]["Average\n(Across)\n(m)(ColIdx:+)"]), label="Avg. Across", marker="o", linestyle='None')
                    ax.set_xticks(range(max_pitch - mod_pitch + 20, min_pitch - mod_pitch, -20))
                    ax.set_yticks(range(max_across - mod_across + 100, min_across - mod_across, -100))
                    ax.grid()
                    ax.set_xlabel("Pitch tilt (deg)")
                    ax.set_ylabel("Avg. Across Error (m)")
                    ax.legend()
                    ax.set_title("Geolocation Error (Without GCP)\nPitch vs Average Across Error (m)")
                    st.pyplot(fig)

                with fig_col8:
                    fig, ax = plt.subplots(figsize=(13, 5))
                    ax.plot(np.array(df["Attitude"]["Pitch\n(deg)"]), np.array(df["Geolocation Error\n(Without GCP)"]["Average\n(Along)\n(m)(LineNo:+)"]), label="Avg. Along", marker="o", linestyle='None')
                    ax.set_xticks(range(max_pitch - mod_pitch + 20, min_pitch - mod_pitch, -20))
                    ax.set_yticks(range(max_along - mod_along + 100, min_along - mod_along, -100))
                    ax.grid()
                    ax.set_xlabel("Pitch tilt (deg)")
                    ax.set_ylabel("Avg. Along Error (m)")
                    ax.legend()
                    ax.set_title("Geolocation Error (Without GCP)\nPitch vs Average Along Error (m)")
                    st.pyplot(fig)

                # Show Summary DataFrame
                cols = ['Num. GCPs', 'CE90\n(m)', 'Average\n(m)',
                        'Average\n(Horizontal)\n(m)(East:+)',
                        'Average\n(Vertical)\n(m)(North:+)',
                        'Std. Dev.\n(Horizontal)\n(m)(East:+)',
                        'Std. Dev.\n(Vertical)\n(m)(North:+)',
                        'Average\n(Across)\n(m)(ColIdx:+)', 'Average\n(Along)\n(m)(LineNo:+)',
                        'Std. Dev.\n(Across)\n(m)', 'Std. Dev.\n(Along)\n(m)']
                rows = ["CE90", "Avg", "Std", "Min", "Max"]
                summarydata = {c: {r: 0 for r in rows} for c in cols}
                for c in cols:
                    sort_data = np.sort(df["Geolocation Error\n(Without GCP)"][c])[::-1]
                    for r in rows:
                        if r == "CE90":
                            value = sort_data[int(np.round(df_len * 0.1 + 0.5))]
                        elif r == "Avg":
                            value = np.nanmean(sort_data)
                        elif r == "Std":
                            value = np.nanstd(sort_data)
                        elif r == "Min":
                            value = np.nanmin(sort_data)
                        else:
                            value = np.nanmax(sort_data)
                        summarydata[c][r] = value

                summarydf = pd.DataFrame(summarydata)
                st.markdown("### Geolocation Accuracy")
                st.dataframe(summarydata)
                time.sleep(1)

                # show Detail DataFrame
                st.markdown("### Full Data")
                st.dataframe(df)
                time.sleep(1)
