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
    "host":"10.164.0.4",
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

def torque_over_time(conn, machine_number: int, tag: Literal["torque1", "torque2"], timerange: tuple[datetime, datetime]):
    cur = conn.cursor()
    cur.execute(SQL_STRING, (machine_number, tag, timerange[0], timerange[1]))
    records = cur.fetchall()
    cur.close()
    return records

def main():
    conn = psycopg2.connect(**PG_CONFIG)
    local_tz = timezone(timedelta(hours=2))

    dataset=torque_over_time(conn, 12, "torque1", (datetime(2025,7,2,9,40,30,tzinfo=local_tz)+timedelta(minutes=10), datetime(2025,7,2,14,29,0,tzinfo=local_tz)-timedelta(minutes=10)))
    #dataset=torque_over_time(conn, 12, "torque1", (datetime(2025,7,3,11,22,0,tzinfo=local_tz)+timedelta(minutes=10), datetime(2025,7,3,14,37,55,tzinfo=local_tz)-timedelta(minutes=10)))
    #dataset=torque_over_time(conn, 12, "torque1", (datetime(2025,7,9,14,54,54, tzinfo=local_tz)+timedelta(minutes=1), datetime(2025,7,9,15,24,40,tzinfo=local_tz)-timedelta(minutes=1)))

    timestamps, _, _, values = zip(*dataset)
    plt.plot(timestamps, [value if value else np.nan for value in values])
    plt.show()


if __name__ == '__main__':
    main()
