import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from scipy.interpolate import griddata
from netCDF4 import Dataset as NetCDFFile
from datetime import datetime, timedelta

def request_to_interpolate_again():
  user_choice = input("Hacer otra interpolación?\n1. Sí\n2. No\nRespuesta: ")

  if user_choice.strip().capitalize() in ["Sí", "Si"] or user_choice == "1":
    measurement = int(input("1. Temperatura\n2. Velocidad del viento\n"
        "3. Dirección del viento\n" 
        "Introduce el número de la medida a interpolar: "
      )
    )

    while (measurement in range(1, 4)):
      spatial_interpolator(measurement)
      measurement = False
  else: 
    print("\nEntendido\n")

longs_to_interp = np.linspace(25.5, 25, 72).reshape(72, 1)
lats_to_interp = np.linspace(66, 69, 72).reshape(72, 1)
coords_to_interp = np.block([longs_to_interp, lats_to_interp])

def temporal_interpolator(interpolants, desired_data, out=None):
  print("\nInterpolación temporal en proceso...\n")
  time_interpolants = []

  desired_dates = desired_data["desired_dates"]
  explosion_dates = desired_data["explosion_dates"]

  array_shape = np.empty(
    interpolants.size // 2
  ).reshape(23, desired_dates.shape[0], 1, 72)

  parsed_explosion_dates = explosion_dates.astype("int64")
  parsed_desired_dates =  desired_dates.astype("int64")

  with np.nditer(
    [array_shape, out], flags=["multi_index", "reduce_ok"], 
    op_flags=[["readonly"], ["readwrite", "allocate"]], 
    op_axes=[[0, 1, 2, 3], None]
  ) as it: 
    it.operands[1][...] = 0
    it.reset()

    for array, temporal_interpolants in it:
      interpolant = np.interp(
        parsed_explosion_dates[it.multi_index[1]],
        parsed_desired_dates[it.multi_index[1], ...],
        [
          interpolants[
            it.multi_index[0], it.multi_index[1], 
            it.multi_index[2], it.multi_index[3]
          ],
          interpolants[
            it.multi_index[0], it.multi_index[1],
            it.multi_index[2] + 1, it.multi_index[3]
          ],
        ]
      )
      temporal_interpolants[...] = interpolant
    result = it.operands[1]
    time_interpolants.append(result)
  array_of_time_interpolants = np.array(time_interpolants)

  print("\nInterpolación temporal exitosa\n")
  return array_of_time_interpolants

def spatial_slice_to_interp(coords, measurement_slice):  
  return griddata(
    coords, measurement_slice.ravel(), coords_to_interp, 
    method="cubic" 
  )

def spatial_interpolator(coords, measurement, desired_data):
  desired_dates = desired_data["desired_dates"]
  desired_indexes = desired_data["desired_indexes"]
  explosion_dates = desired_data["explosion_dates"]
  
  transposed = np.transpose(measurement[desired_indexes, ...], (1, 0, 2, 3))
  height_dim = transposed.shape[0]
  time_dim = desired_dates.shape[0] * 2
  spatial_interpolants = []
    
  with Parallel(n_jobs=-1, prefer="processes", verbose=10) as parallel:
    spatial_interpolants = parallel(
      [
        delayed(spatial_slice_to_interp)
        (coords, transposed[h, t, ...])
        for t in range(time_dim) for h in range(height_dim)
      ]
    )

  spatial_interpolants = np.array(
    spatial_interpolants
  ).reshape(height_dim, desired_dates.shape[0], 2, 72)

  return {
    "interpolants": spatial_interpolants, 
    "desired_data": {
      "desired_dates": desired_dates, "explosion_dates": explosion_dates
    }
  }

def get_desired_indices(explosion_dates, all_dates):
  desired_indices = []
  
  with np.nditer([explosion_dates], op_flags=["readonly"]) as it:
    for explosion_date in it:
      desired_indices.extend(
        np.argsort(np.abs(all_dates - explosion_date))[:2]
      )           
  return desired_indices

def get_explosion_dates(files):
  explosion_dates = np.empty(len(files), dtype="datetime64[m]")

  for index, file in enumerate(files):
    data = pd.read_csv(file, delim_whitespace=True, header=None)

    explosion = np.argmax(data.iloc[:,3])
    e_date = data.iloc[explosion, 0]
    e_year = e_date[0:4]
    e_day = e_date[5:8]
    e_day = e_day.rjust(3 + len(e_day), "0")
    e_hour = e_date[9:11]
    e_min = e_date[12:14]

    template_date = datetime(int(e_year), 1, 1, int(e_hour), int(e_min))
    date_object = template_date + timedelta(days=int(e_day) - 1)
    explosion_dates[index] = date_object.strftime("%Y-%m-%dT%H:%M")
    
  return explosion_dates

def get_desired_data(files, all_dates):
  explosion_dates = get_explosion_dates(files)
  indexes = get_desired_indices(explosion_dates, all_dates)
  desired_dates = np.sort(
    all_dates[indexes]
  ).reshape(all_dates[indexes].size // 2, 2)

  return {
    "desired_indexes": indexes, "explosion_dates": explosion_dates,
    "desired_dates": desired_dates
  }

def load_nc_data(file, user_choice):
  nc = NetCDFFile(file)

  longs = nc.variables["longitude"][100:475, 477:]
  longs = np.where(longs <= 180, longs, longs - 360).reshape(np.size(longs), 1)

  lats = nc.variables["latitude"][100:475, 477:]
  lats = lats.reshape(np.size(lats), 1)

  full_set_of_coords = np.hstack((longs, lats))

  dates = nc.variables["time"][:]
  dates = np.array(
    [
      datetime.utcfromtimestamp(posix_ts).strftime("%Y-%m-%dT%H:%M")
      for posix_ts in dates
    ], 
    dtype="datetime64[m]"
  )

  measurement = nc.variables['t'][..., 100:475, 477:]
  if (user_choice == 2):
    measurement = nc.variables['u'][..., 100:500, 450:]
  elif (user_choice == 3):
    measurement = nc.variables['v'][..., 100:500, 450:]
  
  nc.close()
  return {
    "coords": full_set_of_coords, "dates": dates, "measurement": measurement
  }

input_file = "./carra-files/param_130.nc"
user_choice = int(input("1. Temperatura\n2. Velocidad del viento\n3. Dirección"
" del viento\nIntroduce el número de la medida a interpolar: "))

dir = "./explosion-times"
arc_files = [
  f"{dir}/2016-231_12.29.58.500.ARC", f"{dir}/2016-232_11.29.59.350.ARC", 
	f"{dir}/2016-233_13.29.59.550.ARC", f"{dir}/2016-234_13.00.00.100.ARC", 
  f"{dir}/2016-235_11.59.59.500.ARC", f"{dir}/2016-236_12.59.59.450.ARC", 
	f"{dir}/2016-237_11.59.59.400.ARC", f"{dir}/2016-238_11.29.59.400.ARC", 
  f"{dir}/2016-239_11.29.58.650.ARC", f"{dir}/2016-240_12.59.59.100.ARC", 
  f"{dir}/2016-241_10.59.59.000.ARC", f"{dir}/2016-242_09.59.58.550.ARC", 
  f"{dir}/2016-242_14.09.58.450.ARC", f"{dir}/2016-243_07.54.57.750.ARC", 
  f"{dir}/2016-243_10.04.57.300.ARC", f"{dir}/2016-243_12.24.56.850.ARC", 
  f"{dir}/2016-244_08.49.58.950.ARC", f"{dir}/2016-244_11.44.58.700.ARC", 
  f"{dir}/2016-244_15.44.58.750.ARC"
]

while (user_choice in range(1, 4)):
  nc_data = load_nc_data(input_file, user_choice)
  desired_data = get_desired_data(arc_files, nc_data["dates"])
  spatial_interpolator_d = spatial_interpolator(
    nc_data["coords"], nc_data["measurement"], desired_data
  )
  final_interpolants = temporal_interpolator(
    spatial_interpolator_d["interpolants"], 
    spatial_interpolator_d["desired_data"]
  )
  request_to_interpolate_again()
  user_choice = False