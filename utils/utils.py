# Get Sea Level Anomaly Data Near Lat Long Coords
# General dataset info at:
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-level-global
#
# Usage:
# Create a login and setup account as outlined here:
# https://cds.climate.copernicus.eu/api-how-to

# Then insure cdsapi is installed with :
# conda install -c conda-forge cdsapi
# Ensure ~/.cdsapirc file is configured as outlined above

import cdsapi
import pandas as pd
import numpy as np
import os
import xarray as xr
import shutil
import glob


# Example of chaining functions defined here
def download_process_sea_data():
    get_sea_level_raw(1999, 2000, path_out="data/sea_level/")
    extract_tars(
        file_pattern="*_download.tar.gz",
        path_in="data/sea_level/",
        path_out="data/sea_level/netcdf/",
    )
    build_sea_data(
        start_year=1999,
        end_year=2000,
        netcdf_path="data/sea_level/netcdf/",
        target_lon=175.8606890,
        target_lat=-36.993684,
        buffer_degrees=0.5,
        path_out="data",
    )


def get_sea_level_raw(start_year, end_year, path_out):
    """Gets global sea level data from Copernicus data store between two years inclusive.
    Grabs all months + days for each year, each year is ~ 2.5 GB per tar.gz archive downloaded.

    Parameters
    ----------
    start_year : int
        Year to start grabbing sea level data
    end_year : int
        Last year to grab sea level data
    path_out : str
        File path to save to
    """
    c = cdsapi.Client()

    for year in range(start_year, end_year + 1):

        print(f"Starting Year: {year}")

        c.retrieve(
            "satellite-sea-level-global",
            {
                "format": "tgz",
                "year": [str(year)],
                "month": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                ],
                "day": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                    "16",
                    "17",
                    "18",
                    "19",
                    "20",
                    "21",
                    "22",
                    "23",
                    "24",
                    "25",
                    "26",
                    "27",
                    "28",
                    "29",
                    "30",
                    "31",
                ],
            },
            os.path.join(path_out, str(year) + "_download.tar.gz"),
        )


def extract_tars(file_pattern, path_in, path_out):
    """Extracts .tar.gz file to the NetCDF files

    Parameters
    ----------
    file_pattern : str
        File pattern to identify .tar.gz file of NetCDF files in folder
    path_in : str
        File path to folder of NetCDF files to extract
    path_out : str
        File path to extract NetCDF files to
    """
    for f in glob.glob(os.path.join(path_in, file_pattern)):
        shutil.unpack_archive(f, path_out)


def build_sea_data(
    start_year=1999,
    end_year=2016,
    netcdf_path="data/sea_level/netcdf/",
    target_lon=175.8606890,
    target_lat=-36.993684,
    buffer_degrees=0.5,
    path_out=".",
):
    """Builds a Pandas DataFrame of sea level data near a specific lat/lon pair
    and within a buffer on lat/long degrees (as a poor mans approach to using xarray for now).
    Averages to monthly data, start of each month for:

    adt: sea surface height above geoid (m)
    vgos: absolute geostrophic velocity, zonal component (m/s)
    ugos: absolute geostrophic velocity, meridian component (m/s)

    Parameters
    ----------
    start_year : int, optional
        year to start extracting, by default 1999
    end_year : int, optional
        year to finish extracting, inclusive, by default 2016
    netcdf_path : str, optional
        path to where sea level netCDF files are stored, by default "data/sea_level/netcdf/"
    target_lon : float, optional
        Longitude to target, from 0->360 degrees east, by default 175.8606890
    target_lat : float, optional
        Latitude to target, from -90 -> 90, by default -36.993684
    buffer_degrees : float, optional
        range to average over for sea level data around target lat, long, by default 0.5
    path_out : str, optional
        Path to save out csv file to, by default "."
    """
    # tairua_coords = (-36.993684, 175.8606890)
    df_sea_data = pd.DataFrame()

    for year in range(start_year, end_year + 1):
        ds_first = xr.open_mfdataset(
            os.path.join(netcdf_path, f"dt_global_twosat_phy_l4_{year}*.nc")
        )

        target_lon = xr.DataArray(
            list(target_lon + np.linspace(-buffer_degrees, buffer_degrees))
        )
        target_lat = xr.DataArray(
            list(target_lat + np.linspace(-buffer_degrees, buffer_degrees))
        )

        ds_tairua = ds_first[["adt", "ugos", "vgos"]].sel(
            longitude=target_lon, latitude=target_lat, method="nearest"
        )
        df_sealevel_pandas = (
            ds_tairua.resample(time="MS")
            .mean()
            .mean(dim="dim_0")
            .to_dataframe()
        )

        df_sea_data = pd.concat([df_sea_data, df_sealevel_pandas])

        print(
            f"************************Done {year} ************************************"
        )
        print(df_sea_data.tail(10))

    df_sea_data.to_csv(os.path.join(path_out, "df_sea_data.csv"))
