from .base_agent import BaseAgent
from .ground_agents import SignalDetector, TransportVehicle, RescueVehicle
from .air_agents import UAV
from .space_agents import Satellite
from .fixed_station import FixedStation


def create_agents(config, env):
    agents = []

    for i in range(config['num_signal_detectors']):
        agents.append(SignalDetector(env, config, f"SD_{i}"))

    for i in range(config['num_transport_vehicles']):
        agents.append(TransportVehicle(env, config, f"TV_{i}"))

    for i in range(config['num_rescue_vehicles']):
        agents.append(RescueVehicle(env, config, f"RV_{i}"))

    for i in range(config['num_uavs']):
        agents.append(UAV(env, config, f"UAV_{i}"))

    for i in range(config['num_satellites']):
        agents.append(Satellite(env, config, f"SAT_{i}"))

    for i in range(config['num_fixed_stations']):
        agents.append(FixedStation(env, config, f"FS_{i}"))

    return agents