import multiprocessing
import cdsapi
import os
import argparse

client = cdsapi.Client()

def create_request(year, resolution, pressure_level, variable_id):
    return {
        'product_type': 'reanalysis',
        'format': 'grib',
        'pressure_level': str(pressure_level),
        'year': str(year),
        'time': [f'{str(hour).rjust(2, "0")}:00' for hour in range(0, 24)],
        'day': [str(day) for day in range(1, 32)],
        'month': [str(month) for month in range(1, 13)],
        'variable': variable_id,
        'grid': [resolution, resolution]
    }


def send_request(args):
    year, resolution, pressure_level, variable_id, output_dir, output_prefix = args

    filepath = os.path.join(output_dir, f"{output_prefix}_g{resolution}_{pressure_level}hpa_{year}_data.grib")
    if os.path.exists(filepath):
        print(f"Already downloaded '{filepath}'.")
        return

    request = create_request(year, resolution, pressure_level, variable_id)
    print(f"Sending request for year {year}.")
    client.retrieve('reanalysis-era5-pressure-levels', request, filepath)


def download_variable(start_year, end_year, variable_id, resolution, pressure_level, output_dir, output_prefix, n_jobs):
    years = list(range(start_year, end_year + 1))
    params = [(y, resolution, pressure_level, variable_id, output_dir, output_prefix) for y in years]

    pool = multiprocessing.Pool(processes=n_jobs)
    x = pool.map_async(send_request, params)
    x.get()
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download ERA5 data')
    parser.add_argument('--variable_id', type=str, default='total_precipitation', help='Variable ID')
    parser.add_argument('--start_year', type=int, default=1940, help='Start year')
    parser.add_argument('--end_year', type=int, default=2022, help='End year')
    parser.add_argument('--resolution', type=float, default=0.5, help='Resolution')
    parser.add_argument('--pressure_level', type=int, default=500, help='Pressure level')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--output_prefix', type=str, default='tp', help='Output prefix')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of jobs')
    args = parser.parse_args()

    download_variable(args.start_year, args.end_year, args.variable_id, args.resolution, args.pressure_level,
                      args.output_dir, args.output_prefix, args.n_jobs)
