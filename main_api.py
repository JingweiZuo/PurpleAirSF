import configparser
from urllib.parse import urljoin
from io import BytesIO

import pandas as pd


class PurpleAir(ApiCall):

    def __init__(self, config):
        super().__init__("https://api.purpleair.com/v1/")
        self._config = config
        self._params.update({
            'api_key': config['purpleair.com']['API_readKey']
        })

        self._sensors_api = urljoin(self._api, "sensors")
        self._sensors_by_id_api = urljoin(self._api, "sensors/{sensor_index}/")
        self._csv_api = urljoin(self._sensors_by_id_api, "history/csv")

    def sensor_by_id(self, sensor_index):
        return self.get(self._sensors_by_id_api.format(sensor_index=sensor_index), params=self._params)

    def sensors(self,
                fields='name,latitude,longitude,altitude',
                nwlat=None, nwlng=None, selat=None, selng=None
                ):
        params = self._params | {
            'nwlat': nwlat, 'nwlng': nwlng, 'selat': selat, 'selng': selng,
            'fields': fields,
        }
        return self.get(self._sensors_api, params=params)

    @staticmethod
    def csv_limits(average):
        hour = 60
        day = 24 * hour
        year = 365 * day
        limits = {  # minutes: limit
            0: 2 * 60,  # (real-time)
            10: 3 * day,
            30: 7 * day,
            hour: 14 * day,
            6 * hour: 90 * day,
            day: year,
        }
        if average not in limits:
            raise ValueError(f"Average value of {average} minutes is not available.")
        return limits[average]

    def csv(self,
            sensor_index,
            start_timestamp,
            end_timestamp,
            average=10,  # 0 (real-time), 10 (default if not specified), 30, 60, 360 (6 hour), 1440 (1 day)
            # 'error': 'InvalidTimestampSpanError', 'description':
            # '10 minute average history maximum time span is three (3) days.'
            # 'One hour average history maximum time span is fourteen (14) days.'
            # 'Six hour average history maximum time span is ninety (90) days.'
            # 'One day average history maximum time span is one (1) year.'
            fields=(
                'humidity,temperature,pressure,'
                'voc,'
                'pm1.0_atm,pm1.0_cf_1,'
                'pm2.5_alt,pm2.5_atm,pm2.5_cf_1,'
                'pm10.0_atm,pm10.0_cf_1,'
                'scattering_coefficient,deciviews,visual_range,'
                '0.3_um_count,0.5_um_count,1.0_um_count,2.5_um_count,5.0_um_count,10.0_um_count'
            ),
            # average=None,  # default 10
            params=None) -> pd.DataFrame:
        params = self._params | {
            'start_timestamp': int(start_timestamp),
            'end_timestamp': int(end_timestamp),
            'average': average,
            'fields': fields,
        }
        resp = self.get(self._csv_api.format(sensor_index=sensor_index), params=params, json_=False)
        # print(resp)
        # pp(resp.content)
        csv_file_buffer = BytesIO(resp.content)
        df = pd.read_csv(csv_file_buffer)
        df.time_stamp = pd.to_datetime(df.time_stamp, unit='s', utc=True)
        df.set_index('time_stamp', inplace=True)
        df.sort_index(inplace=True)
        return df


# import configparser
# from main_api import PurpleAir
def get_purpleair_geo_coords(sensor_index):
    """Returns latitude and longitude of a given sensor index"""
    sensor_current_data = purple_air.sensor_by_id(sensor_index)
    return sensor_current_data['sensor']['latitude'], sensor_current_data['sensor']['longitude']


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('keys/PurpleAir_API_key.conf')

    print_json = ApiCall.formatted_print

    # PurpleAir
    purple_air = PurpleAir(config)

    sensor_in_london = purple_air.sensor_by_id(123275)
    print_json(sensor_in_london)
    ts = pd.to_datetime(sensor_in_london['time_stamp'], unit='s')  # utc=True)
    print(ts.tz_localize('Europe/London').tz_convert('Asia/Dubai'))
    assert abs(ts.tz_localize('UTC') - pd.Timestamp.now().tz_localize('Asia/Dubai')) < pd.Timedelta(minutes=1)
    assert abs(ts.tz_localize('UTC') - pd.Timestamp.utcnow()) < pd.Timedelta(minutes=1)

    sensor_in_rome = purple_air.sensor_by_id(169749)
    ts = pd.to_datetime(sensor_in_rome['time_stamp'], unit='s')  # utc=True)
    print(ts.tz_localize('UTC').tz_convert('Asia/Dubai'))
    assert not abs(ts.tz_localize('Europe/Rome') - pd.Timestamp.now().tz_localize('Asia/Dubai')) < pd.Timedelta(minutes=1)
    assert not abs(ts.tz_localize('Europe/Rome') - pd.Timestamp.utcnow()) < pd.Timedelta(minutes=1)
    assert abs(ts.tz_localize('UTC') - pd.Timestamp.now().tz_localize('Asia/Dubai')) < pd.Timedelta(minutes=1)
    assert abs(ts.tz_localize('UTC') - pd.Timestamp.utcnow()) < pd.Timedelta(minutes=1)

    london_sensors = purple_air.sensors(nwlat=51.6000, nwlng=-0.1800, selat=51.4600, selng=-0.0300)
    print_json(london_sensors)

    utcnow = pd.Timestamp.utcnow()
    df = purple_air.csv(146146,
                        (utcnow - pd.Timedelta(weeks=2)).timestamp(),
                        (utcnow - pd.Timedelta(weeks=1)).timestamp(),
                        average=60)
    print(df)
    df = purple_air.csv(146146,
                        (utcnow - pd.Timedelta(days=365*3)).timestamp(),
                        (utcnow - pd.Timedelta(days=365*2)).timestamp(),
                        average=1440)
    print(df)
    df = purple_air.csv(146146,
                        (utcnow - pd.Timedelta(days=2)).timestamp(),
                        (utcnow).timestamp(),
                        average=10)
    print(df)
    df = purple_air.csv(146146,
                        (utcnow - pd.Timedelta(weeks=2)).timestamp(),
                        (utcnow - pd.Timedelta(weeks=1)).timestamp(),
                        average=1440)
    print(df)


