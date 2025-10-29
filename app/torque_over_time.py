from typing import Literal
import matplotlib.pyplot as plt
import matplotlib
import psycopg2
import numpy as np
from datetime import datetime, timezone, timedelta
from pprint import pprint
import csv

matplotlib.use('WebAgg')

PG_CONFIG = {
    "host":"34.13.178.206",
    "password":"DigiRomMake",
    "user": "readonly",
    "dbname": "makeathon"
}


SQL_STRING="""
select ts, machine_number, tag, value_dec
from extruderlift
where machine_number = %s
and tag=%s
and ts between %s and %s
order by ts asc;
"""

SQL_STRING_WITH_GAPFILL="""
select
time_bucket_gapfill('500 ms', ts) as tsbucket,
machine_number,
tag,
locf(max(value_dec)) as value
from extruderlift
where machine_number = %s
and tag=%s
and ts between %s and %s 
group by tsbucket, machine_number, tag
order by tsbucket asc
"""

def torque_over_time(conn, machine_number: int, tag: Literal["torque1", "torque2"], timerange: tuple[datetime, datetime]):
    cur = conn.cursor()
    cur.execute(SQL_STRING_WITH_GAPFILL, (machine_number, tag, timerange[0], timerange[1]))
    records = cur.fetchall()
    cur.close()
    return records


def get_active_windows_for_machine(conn, machine_number, min_duration = '60min'):
    # get all time windows representing an active cycle for a machine
    SQL = f"""
select ts as start_ts, next_ts as stop_ts, next_ts-ts as duration, state  from (
select ts,
  value_str as state,
  LEAD(value_str, 1) over (order by ts asc) as next_state,
  LEAD(ts, 1) over (order by ts asc) as next_ts
from (
select ts, value_str
from extruderlift
where machine_number = {machine_number}
and tag = 'state'
order by ts asc
) as state_changes
) as state_ranges
where state = 'active'
and next_ts-ts > '{min_duration}'
"""
    cur = conn.cursor()
    cur.execute(SQL)
    records = cur.fetchall()
    cur.close()
    return records


def write_csv(machine_number, start, end, size, dataset):
    # write out csv file of size $size for given dataset if it contains enough points
    if len(dataset) < size:
        return False

    dataset = dataset[0:size]

    with open(f"machine_{machine_number}_torque1_{size}_values_{start.isoformat()}-{end.isoformat()}.csv", "w") as dataf:
            writer = csv.writer(dataf, delimiter=";")
            writer.writerow(["timestamp", "machine_number", "tag", "value_dec"])
            writer.writerows([value for value in dataset if value[3] is not None])

    return True



def generate_torque_data_for_time_windows(conn, machine_number):
    # walk all active windows of a machine and generate csv files for all windows
    windows = get_active_windows_for_machine(conn, machine_number) 
    for index, window in enumerate(windows):
        print(f"Processing active window {index} of {len(windows)} for machine {machine_number}")
        dataset = torque_over_time(conn, machine_number, "torque1", (window[0], window[1]))
        print(len(dataset))

        if (write_csv(machine_number, window[0], window[1], 10000, dataset)):
            print("Generated CSV with 10000 datapoints")

        if (write_csv(machine_number, window[0], window[1], 100000, dataset)):
            print("Generated CSV with 100000 datapoints")

    
def main():
    conn = psycopg2.connect(**PG_CONFIG)
    local_tz = timezone(timedelta(hours=2))

    # generate csv files for all active time windows
    for machine_number in range(1,15):
        print(f"generating csv files for machine {machine_number}")
        generate_torque_data_for_time_windows(conn, machine_number)


if __name__ == '__main__':
    main()
