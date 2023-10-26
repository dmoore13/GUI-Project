import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.interpolate import interp1d
import math

#region All of the data
unused_oil_table = np.array([
    [0, 32, 0.899, 1796, 0.429, 42.8e-4, 46.1, 0.147, 0.085, 9.11, 3.53, 47100],
    [20, 68, 0.888, 1880, 0.449, 9.0e-4, 9.7, 0.145, 0.084, 8.72, 3.38, 10400],
    [40, 104, 0.876, 1964, 0.469, 2.4e-4, 2.6, 0.144, 0.083, 8.34, 3.23, 2870],
    [60, 140, 0.864, 2047, 0.489, 0.839e-4, 0.903, 0.140, 0.081, 8.00, 3.10, 1050],
    [80, 176, 0.852, 2131, 0.509, 0.375e-4, 0.404, 0.138, 0.080, 7.69, 2.98, 490],
    [100, 212, 0.840, 2219, 0.530, 0.203e-4, 0.219, 0.137, 0.079, 7.38, 2.86, 276],
    [120, 248, 0.828, 2307, 0.551, 0.124e-4, 0.133, 0.135, 0.078, 7.10, 2.75, 175],
    [140, 284, 0.816, 2395, 0.572, 0.080e-4, 0.086, 0.133, 0.077, 6.86, 2.66, 116],
    [160, 320, 0.805, 2483, 0.593, 0.056e-4, 0.060, 0.132, 0.076, 6.63, 2.57, 84]
])
ethylene_glycol_table = np.array([
    [0, 32, 1.130, 2294, 0.548, 57.53e-6, 61.92, 0.242, 0.140, 9.34, 3.62, 615],
    [20, 68, 1.116, 2382, 0.569, 19.18e-6, 20.64, 0.249, 0.144, 9.39, 3.64, 204],
    [40, 104, 1.101, 2474, 0.591, 8.69e-6, 9.35, 0.256, 0.148, 9.39, 3.64, 93],
    [60, 140, 1.087, 2562, 0.612, 4.75e-6, 5.11, 0.260, 0.150, 9.32, 3.61, 51],
    [80, 176, 1.077, 2650, 0.633, 2.98e-6, 3.21, 0.261, 0.151, 9.21, 3.57, 32.4],
    [100, 212, 1.058, 2742, 0.655, 2.03e-6, 2.18, 0.263, 0.152, 9.08, 3.52, 22.4]
])
water_table = np.array([
    [0, 32, 1.002, 4217, 1.0074, 17.88e-7, 19.25, 0.552, 0.319, 1.308, 5.07, 13.6],
    [20, 68, 1.000, 4181, 0.9988, 10.06e-7, 10.83, 0.597, 0.345, 1.430, 5.54, 7.02],
    [40, 104, 0.994, 4187, 0.9970, 6.58e-7, 7.05, 0.628, 0.363, 1.512, 5.86, 4.32],
    [60, 140, 0.985, 4184, 0.9994, 4.78e-7, 5.14, 0.651, 0.376, 1.554, 6.02, 3.02],
    [80, 176, 0.974, 4196, 1.0023, 3.64e-7, 3.92, 0.668, 0.386, 1.636, 6.34, 2.22],
    [100, 212, 0.960, 4196, 1.0070, 2.94e-7, 3.16, 0.680, 0.393, 1.680, 6.51, 1.74],
    [120, 248, 0.945, 4250, 1.015, 2.47e-7, 2.66, 0.685, 0.396, 1.708, 6.62, 1.446],
    [140, 284, 0.928, 4283, 1.023, 2.14e-7, 2.30, 0.684, 0.395, 1.724, 6.68, 1.241],
    [160, 320, 0.909, 4334, 1.037, 1.90e-7, 2.04, 0.670, 0.390, 1.729, 6.70, 1.099],
    [180, 356, 0.889, 4417, 1.055, 1.73e-7, 1.86, 0.675, 0.390, 1.724, 6.68, 1.004],
    [200, 392, 0.866, 4505, 1.076, 1.60e-7, 1.72, 0.665, 0.384, 1.706, 6.61, 0.937],
    [220, 428, 0.842, 4585, 1.101, 1.50e-7, 1.54, 0.572, 0.377, 1.680, 6.51, 0.891],
    [240, 464, 0.815, 4679, 1.136, 1.43e-7, 1.54, 0.635, 0.369, 1.639, 6.35, 0.879],
    [260, 500, 0.785, 4769, 1.174, 1.37e-7, 1.54, 0.611, 0.354, 1.577, 6.11, 0.874],
    [280, 537, 0.752, 5088, 1.244, 1.35e-7, 1.54, 0.540, 0.312, 1.324, 5.13, 1.109],
    [300, 572, 0.714, 5728, 1.368, 1.35e-7, 1.45, 0.540, 0.312, 1.324, 5.13, 1.109]
])

tables = {
    "Unused Oil": unused_oil_table,
    "Ethylene Glycol": ethylene_glycol_table,
    "Water": water_table
}

columns = ["Temp °C", "Temp °F", "Density kg/m^3", "Specific Heat Cp J_kg.K", "Specific Heat Cp lbm-°R",
           "Kinematic Viscosity v m^2/s x 10^4", "Kinematic Viscosity v ft^2/s x 10^3", "Thermal Conductivity k W_m.K",
           "Thermal Conductivity k BTU_hr.ft-°R", "Thermal Diffusivity α m^2/s x 10^8",
           "Thermal Diffusivity α ft^2/hr x 10^3",
           "Prandtl Number Pr"]

pipe_data = {
    '1/8': {
        '40 (std)': {'Outside Diameter': 1.029, 'Inside Diameter': 0.683, 'Flow Area': 0.3664},
        '80 (xs)': {'Outside Diameter': 1.029, 'Inside Diameter': 0.547, 'Flow Area': 0.235},
    },
    '1/4': {
        '40 (std)': {'Outside Diameter': 1.372, 'Inside Diameter': 0.924, 'Flow Area': 0.670},
        '80 (xs)': {'Outside Diameter': 1.372, 'Inside Diameter': 0.768, 'Flow Area': 0.463},
    },
    '3/8': {
        '40 (std)': {'Outside Diameter': 1.714, 'Inside Diameter': 1.252, 'Flow Area': 1.233},
        '80 (xs)': {'Outside Diameter': 1.714, 'Inside Diameter': 1.074, 'Flow Area': 0.9059},
    },
    '1/2': {
        '40 (std)': {'Outside Diameter': 2.134, 'Inside Diameter': 1.580, 'Flow Area': 1.961},
        '80 (xs)': {'Outside Diameter': 2.134, 'Inside Diameter': 1.386, 'Flow Area': 1.508},
        '160 ()': {'Outside Diameter': 2.134, 'Inside Diameter': 1.178, 'Flow Area': 1.090},
        ' (xxs)': {'Outside Diameter': 2.134, 'Inside Diameter': 0.640, 'Flow Area': 0.32},
    },
    '3/4': {
        '40 (std)': {'Outside Diameter': 2.667, 'Inside Diameter': 2.093, 'Flow Area': 3.441},
        '80 (xs)': {'Outside Diameter': 2.667, 'Inside Diameter': 1.883, 'Flow Area': 1.895},
        '160 ()': {'Outside Diameter': 2.667, 'Inside Diameter': 1.555, 'Flow Area': 1.898},
        ' (xxs)': {'Outside Diameter': 2.667, 'Inside Diameter': 1.103, 'Flow Area': 0.955}
    },
    '1': {
        '40 (std)': {'Outside Diameter': 3.340, 'Inside Diameter': 2.664, 'Flow Area': 5.574},
        '80 (xs)': {'Outside Diameter': 3.340, 'Inside Diameter': 2.430, 'Flow Area': 5.083},
        '160 ()': {'Outside Diameter': 3.340, 'Inside Diameter': 2.070, 'Flow Area': 3.365},
        ' (xxs)': {'Outside Diameter': 3.340, 'Inside Diameter':  1.522, 'Flow Area': 1.815}
    },
    '1 1/4': {
        #'40 (std)': {'Outside Diameter': 4.216, 'Inside Diameter': 3.504, 'Flow Area': 9.643},
        '40 (std)': {'Outside Diameter': 3.493, 'Inside Diameter': 3.28, 'Flow Area': 9.643},
        '80 (xs)': {'Outside Diameter': 4.216, 'Inside Diameter': 3.246, 'Flow Area': 8.275},
        '160 ()': {'Outside Diameter':  4.216, 'Inside Diameter': 2.946, 'Flow Area': 6.816},
        ' (xxs)': {'Outside Diameter': 4.216, 'Inside Diameter': 2.276, 'Flow Area': 4.069}
    },
    '1 1/2': {
        '40 (std)': {'Outside Diameter': 4.826, 'Inside Diameter': 4.090, 'Flow Area': 13.13},
        '80 (xs)': {'Outside Diameter': 4.826, 'Inside Diameter': 3.810, 'Flow Area': 11.40},
        '160 ()': {'Outside Diameter': 4.826, 'Inside Diameter': 3.398, 'Flow Area': 9.068},
        ' (xxs)': {'Outside Diameter': 4.826, 'Inside Diameter': 2.794, 'Flow Area': 6.13}
    },
    '2': {
        #'40 (std)': {'Outside Diameter': 6.034, 'Inside Diameter': 5.252, 'Flow Area': 21.66},
        '40 (std)': {'Outside Diameter': 6.034, 'Inside Diameter': 5.1, 'Flow Area': 21.66},
        '80 (xs)': {'Outside Diameter': 6.034, 'Inside Diameter': 4.926, 'Flow Area': 19.06},
        '160 ()': {'Outside Diameter': 6.034, 'Inside Diameter': 4.286, 'Flow Area': 14.43},
        ' (xxs)': {'Outside Diameter': 6.034, 'Inside Diameter': 3.820, 'Flow Area': 11.46},
    },
    '2 1/2': {
        '40 (std)': {'Outside Diameter': 7.303, 'Inside Diameter': 6.271, 'Flow Area': 30.89},
        '80 (xs)': {'Outside Diameter': 7.303, 'Inside Diameter': 5.901, 'Flow Area': 27.35},
        '160 ()': {'Outside Diameter': 7.303, 'Inside Diameter': 5.397, 'Flow Area': 22.88},
        ' (xxs)': {'Outside Diameter': 7.303, 'Inside Diameter': 4.499, 'Flow Area': 15.90},
    },
    '3': {
        '40 (std)': {'Outside Diameter': 8.890, 'Inside Diameter': 7.792, 'Flow Area': 47.69},
        '80 (xs)': {'Outside Diameter': 8.890, 'Inside Diameter': 7.366, 'Flow Area': 42.61},
        '160 ()': {'Outside Diameter': 8.890, 'Inside Diameter': 6.664, 'Flow Area': 34.88},
        ' (xxs)': {'Outside Diameter': 8.890, 'Inside Diameter': 5.842, 'Flow Area': 26.80},
    },
    '3 1/2': {
        '40 (std)': {'Outside Diameter': 10.16, 'Inside Diameter': 9.012, 'Flow Area': 63.79},
        '80 (xs)': {'Outside Diameter': 10.16, 'Inside Diameter': 8.544, 'Flow Area': 57.33},
    },
    '4': {
        '40 (std)': {'Outside Diameter': 11.43, 'Inside Diameter': 10.23, 'Flow Area': 82.19},
        '80 (xs)': {'Outside Diameter': 11.43, 'Inside Diameter': 9.718, 'Flow Area': 74.17},
        '120 ()': {'Outside Diameter': 11.43, 'Inside Diameter': 9.204, 'Flow Area': 66.54},
        '160 ()': {'Outside Diameter': 11.43, 'Inside Diameter': 8.732, 'Flow Area': 59.88},
        ' (xxs)': {'Outside Diameter': 11.43, 'Inside Diameter': 8.006, 'Flow Area': 50.34},
    },
    '5': {
        '40 (std)': {'Outside Diameter': 14.13, 'Inside Diameter': 12.82, 'Flow Area': 129.10},
        '80 (xs)': {'Outside Diameter': 14.13, 'Inside Diameter': 12.22, 'Flow Area': 117.30},
        '120 ()': {'Outside Diameter': 14.13, 'Inside Diameter': 11.59, 'Flow Area': 105.50},
        '160 ()': {'Outside Diameter': 14.13, 'Inside Diameter': 10.95, 'Flow Area': 94.17},
        ' (xxs)': {'Outside Diameter': 14.13, 'Inside Diameter': 10.32, 'Flow Area': 83.65},
    },
    '6': {
        '40 (std)': {'Outside Diameter': 16.83, 'Inside Diameter': 15.41, 'Flow Area': 186.50},
        '80 (xs)': {'Outside Diameter':  16.83, 'Inside Diameter': 14.64, 'Flow Area': 168.30},
        '120 ()': {'Outside Diameter': 16.83, 'Inside Diameter': 13.98, 'Flow Area': 153.50},
        '160 ()': {'Outside Diameter': 16.83, 'Inside Diameter': 13.18, 'Flow Area': 136.40},
        ' (xxs)': {'Outside Diameter': 16.83, 'Inside Diameter': 12.44, 'Flow Area': 121.50},
    }

    # ... more sizes can be added similarly
}
#endregion

#region All of the functions for calculations
def interpolate_table(table, input_temp, temp_unit):
    if temp_unit == "Fahrenheit":
        temp_column = 1
    else:
        temp_column = 0

    input_values = []
    for i in range(1, len(columns)):
        f = interp1d(table[:, temp_column], table[:, i], kind='linear', fill_value="extrapolate")
        interpolated_value = f(input_temp)

        if i == 2:  # Density column (Specific Gravity * 1000)
            interpolated_value *= 1000

        input_values.append(interpolated_value)

    return input_values

def calculate_values():
    fluid1 = fluid_var1.get() #annulus
    temp1 = float(temp_entry1.get())
    unit1 = unit_var1.get()

    fluid2 = fluid_var2.get() #inner pipe
    temp2 = float(temp_entry2.get())
    unit2 = unit_var2.get()

    if unit1 == "Fahrenheit":
        temp1 = (temp1 - 32) * (5 / 9)
    if unit2 == "Fahrenheit":
        temp2 = (temp2 - 32) * (5 / 9)

    average_temp = (temp1 + temp2) / 2

    table1 = tables[fluid1]
    table2 = tables[fluid2]

    interpolated_values1 = interpolate_table(table1, average_temp, "Celsius")
    interpolated_values2 = interpolate_table(table2, average_temp, "Celsius")

    results_dict1 = {columns[i + 1]: val for i, val in enumerate(interpolated_values1)}
    results_dict2 = {columns[i + 1]: val for i, val in enumerate(interpolated_values2)}


    density1SI = results_dict1.get("Density kg/m^3")
    specific_heat1SI =results_dict1.get("Specific Heat Cp J_kg.K")
    kinematic_visc1SI = results_dict1.get("Kinematic Viscosity v m^2/s x 10^4")
    thermal_cond1SI = results_dict1.get("Thermal Conductivity k W_m.K")
    thermal_diff1SI = results_dict1.get( "Thermal Diffusivity α m^2/s x 10^8")
    prandtl1 = results_dict1.get("Prandtl Number Pr")

    density2SI = results_dict2.get("Density kg/m^3")
    specific_heat2SI = results_dict2.get("Specific Heat Cp J_kg.K")
    kinematic_visc2SI = results_dict2.get("Kinematic Viscosity v m^2/s x 10^4")
    thermal_cond2SI = results_dict2.get("Thermal Conductivity k W_m.K")
    thermal_diff2SI = results_dict2.get("Thermal Diffusivity α m^2/s x 10^8")
    prandtl2 = results_dict2.get("Prandtl Number Pr")

    print(kinematic_visc2SI, kinematic_visc1SI) #inner_pipe (water), annulus (ethylene glycol)
    return density1SI, specific_heat1SI, kinematic_visc1SI, thermal_diff1SI, thermal_cond1SI, prandtl1, density2SI, specific_heat2SI, kinematic_visc2SI, thermal_diff2SI, thermal_cond2SI, prandtl2

def update_schedule_options(frame_var, dropdown, *args):
    nominal_diameter = frame_var.get()
    if nominal_diameter in pipe_data:
        schedules = list(pipe_data[nominal_diameter].keys())
        dropdown['values'] = [''] + schedules
    else:
        dropdown['values'] = ['']

def fetch_data(frame_type, frame_var, schedule_var):
    nominal_diameter = frame_var.get()
    schedule = schedule_var.get()

    try:
        data = pipe_data[nominal_diameter][schedule]

        if frame_type == "annulus":
            annulus_dict = {
                "Outside Diameter (m)": data['Outside Diameter'] / 100,
                "Inside Diameter (m)": data['Inside Diameter'] / 100,
                "Flow Area (m^2)": data['Flow Area'] / 10000
            }

            return annulus_dict
        elif frame_type == "inner_pipe":
            inner_pipe_dict = {
                "Outside Diameter (m)": data['Outside Diameter'] / 100,
                "Inside Diameter (m)": data['Inside Diameter'] / 100,
                "Flow Area (m^2)": data['Flow Area'] / 10000
            }

            return inner_pipe_dict
        else:
            raise ValueError("Invalid frame type")

    except KeyError:
        return {"error": "Invalid selection. Please choose appropriate values."}

def calculate_flow_areas(inner_pipe_data, annulus_data):

    annulus_OD = annulus_data.get('Outside Diameter (m)')
    annulus_ID = annulus_data.get('Inside Diameter (m)')
    inner_pipe_OD = inner_pipe_data.get('Outside Diameter (m)')
    inner_pipe_ID = inner_pipe_data.get('Inside Diameter (m)')

    print( annulus_OD, annulus_ID, inner_pipe_OD, inner_pipe_ID)

    annulus_FlowArea = 0.25 * math.pi * (annulus_ID ** 2 - inner_pipe_OD ** 2)
    inner_pipe_FlowArea = 0.25*math.pi*inner_pipe_ID**2

    #Annulus Equivalent Diameters
    Dh = annulus_ID-annulus_OD
    De = (annulus_ID**2-inner_pipe_OD**2)/inner_pipe_OD
    print(annulus_FlowArea, inner_pipe_FlowArea)
    return annulus_FlowArea, inner_pipe_FlowArea, Dh, De, inner_pipe_ID, inner_pipe_OD

def calculate_fluid_velocities():
    # Get mass flow rates from the entry boxes
    mass_flow_rate_annulus = float(mass_flow_rate_entry1.get())
    mass_flow_rate_inner_pipe = float(mass_flow_rate_entry2.get())

    # Fetch the flow areas
    annulus_data = fetch_data("annulus", annulus_diameter_var, annulus_schedule_var)
    inner_pipe_data = fetch_data("inner_pipe", inner_diameter_var, inner_schedule_var)
    annulus_FlowArea, inner_pipe_FlowArea, Dh, De, inner_pipe_ID, inner_pipe_OD = calculate_flow_areas(inner_pipe_data, annulus_data)

    # Fetch the densities
    density1SI, specific_heat1SI, kinematic_visc1SI, thermal_diff1SI, thermal_cond1SI, prandtl1, density2SI, specific_heat2SI, kinematic_visc2SI, thermal_diff2SI, thermal_cond2SI, prandtl2 = calculate_values()

    # Calculate the fluid velocities
    velocity_annulus = mass_flow_rate_annulus / (density1SI * annulus_FlowArea)
    velocity_inner_pipe = mass_flow_rate_inner_pipe / (density2SI * inner_pipe_FlowArea)

    return velocity_annulus, velocity_inner_pipe

def calc_Re_Nu():
    # Fetch velocities
    velocity_annulus, velocity_inner_pipe = calculate_fluid_velocities()

    # Fetch kinematic viscosities and Prandtl numbers
    density1SI, _, kinematic_visc1SI, _, _, prandtl1, density2SI, _, kinematic_visc2SI, _, _, prandtl2 = calculate_values()

    # Fetch equivalent diameters
    _, _, Dh, De, inner_pipe_ID, inner_pipe_OD = calculate_flow_areas(fetch_data("inner_pipe", inner_diameter_var, inner_schedule_var), fetch_data("annulus", annulus_diameter_var, annulus_schedule_var))

    # Calculate Reynolds numbers
    print("Annulus velocity expecting 3.19:", velocity_annulus)
    print("De (0.396):", De)
    print("Kinematic (expecting 5.62e-6):",kinematic_visc1SI)
    Re_annulus = (velocity_annulus * De) / kinematic_visc1SI
    print("Re_annulus:", Re_annulus)
    #1SI is ethylene glycol 5.62e-6. ethylene glycol should be in the annulus
    #2SI is water 5.15e-7. water is in the inner tube
    print("inner pipe velocity expecting 0.756:", velocity_inner_pipe)
    print("Inner diamter of inner pipe, expecting 32.8 mm:", inner_pipe_ID)
    print()
    Re_inner_pipe = (velocity_inner_pipe * inner_pipe_ID) / kinematic_visc2SI
    print("Re_inner_pipe:", Re_inner_pipe)

    # Calculate Nusselt numbers using empirical correlation for turbulent flow in pipes
    Nu_inner_pipe = 0.023 * (Re_inner_pipe**0.8) * (prandtl2**0.3)
    Nu_annulus = 0.023 * (Re_annulus ** 0.8) * (prandtl1 ** 0.4)

    print(Re_annulus, Nu_annulus, Re_inner_pipe, Nu_inner_pipe)
    return Re_annulus, Nu_annulus, Re_inner_pipe, Nu_inner_pipe

def calculate_conv_coeff():
    # Fetching Nu values, and other parameters
    _, Nu_annulus, _, Nu_inner_pipe = calc_Re_Nu()
    _, _, _, De, inner_pipe_ID, inner_pipe_OD = calculate_flow_areas(fetch_data("inner_pipe", inner_diameter_var, inner_schedule_var), fetch_data("annulus", annulus_diameter_var, annulus_schedule_var))
    _, _, _, _, thermal_cond1SI, _, _, _, _, _, thermal_cond2SI, _ = calculate_values()

    # Calculating convective heat transfer coefficient for annulus
    h_annulus = Nu_annulus * thermal_cond1SI / De

    # Calculating convective heat transfer coefficient for inner pipe
    h_inner_pipe_i = Nu_inner_pipe * thermal_cond2SI / inner_pipe_ID
    h_inner_pipe_p = h_inner_pipe_i*(inner_pipe_ID/inner_pipe_OD)
    print("hi of inner pipe, 3585:", h_inner_pipe_i)
    print("hp of inner pipe, 3366:", h_inner_pipe_p)
    print("ha of annulus, 2357:", h_annulus)
    return h_annulus, h_inner_pipe_i, h_inner_pipe_p

def exchanger_coefficient():
    # Fetch the convective heat transfer coefficients
    h_annulus, _, h_inner_pipe_p = calculate_conv_coeff()

    # Calculate the inverse of the overall heat transfer coefficient
    Uo_inv = 1/h_inner_pipe_p + 1/h_annulus

    # Calculate the overall heat transfer coefficient
    Uo = 1/Uo_inv

    print("Overall heat transfer coefficient, Uo:", Uo)

    return Uo

def calculate_outlet_temps():
    # Get the specific heats and densities
    (_, specific_heat1SI, _, _, _, _, _, specific_heat2SI, _, _, _, _) = calculate_values()

    # Get the flow areas and diameters
    _, _, _, _, _, inner_pipe_OD = calculate_flow_areas(fetch_data("inner_pipe", inner_diameter_var, inner_schedule_var), fetch_data("annulus", annulus_diameter_var, annulus_schedule_var))

    # Retrieve mass flow rates and exchanger length from the interface
    mass_flow_annulus = float(mass_flow_rate_entry1.get())
    mass_flow_inner_pipe = float(mass_flow_rate_entry2.get())
    exchanger_length = float(exchanger_length_entry.get())

    # Retrieve Uo
    Uo = exchanger_coefficient()

    # Calculate R
    R = (mass_flow_annulus * specific_heat1SI) / (mass_flow_inner_pipe * specific_heat2SI)
    print("R:", R)

    # Calculate A_o
    A_o = math.pi * inner_pipe_OD * exchanger_length
    print("Ao:", A_o)

    # Check the flow type and perform calculations accordingly
    current_flow_type = flow_type.get()

    # Get temperatures from the entry boxes
    t1_annulus = float(temp_entry1.get())
    T1_inner_pipe = float(temp_entry2.get())

    # Convert temperatures if they are in Fahrenheit
    if unit_var1.get() == "Fahrenheit":
        t1_annulus = (t1_annulus - 32) * 5.0 / 9.0  # Convert Fahrenheit to Celsius

    if unit_var2.get() == "Fahrenheit":
        T1_inner_pipe = (T1_inner_pipe - 32) * 5.0 / 9.0  # Convert Fahrenheit to Celsius

    if current_flow_type == "Counterflow":
        # Counterflow calculations
        print("Performing Counterflow calculations...")
        # Placeholder formulas for Counterflow
        E_counter = math.exp(Uo * A_o * (R - 1) / (mass_flow_annulus * specific_heat1SI))
        print(E_counter)
        print(R)
        print(T1_inner_pipe, t1_annulus)
        T2_inner_pipe =  ((T1_inner_pipe * (R - 1)) - (R * t1_annulus * (1 - E_counter))) / (R*E_counter - 1)
        t2_annulus = t1_annulus + (T1_inner_pipe - T2_inner_pipe) / R

    elif current_flow_type == "Parallel Flow":
        # Parallel Flow calculations
        print("Performing Parallel Flow calculations...")
        E_para = math.exp(Uo * A_o * (R + 1) / (mass_flow_annulus * specific_heat1SI))

        T2_inner_pipe = ((R + E_para)*T1_inner_pipe + R*t1_annulus*(E_para-1)) / (E_para*(R + 1))
        t2_annulus = t1_annulus + (T1_inner_pipe - T2_inner_pipe) / R

    else:
        raise ValueError(f"Unexpected flow type: {current_flow_type}")
    print(t2_annulus, T2_inner_pipe)
    return t2_annulus, T2_inner_pipe

def calculate_hx_rate():
    # Retrieve mass flow rates and specific heats
    mass_flow_annulus = float(mass_flow_rate_entry1.get())
    (_, specific_heat1SI, _, _, _, _, _, specific_heat2SI, _, _, _, _) = calculate_values()

    # Get temperatures from the entry boxes and convert if necessary
    t1_annulus = float(temp_entry1.get())
    if unit_var1.get() == "Fahrenheit":
        t1_annulus = (t1_annulus - 32) * 5.0 / 9.0  # Convert Fahrenheit to Celsius

    # Calculate outlet temperature for annulus
    t2_annulus, _ = calculate_outlet_temps()

    # Calculate q_hx
    q_hx = mass_flow_annulus * specific_heat1SI * (t2_annulus - t1_annulus)

    return q_hx/1000


#endregion

def on_calculate_click():
    # First command
    calculate_values()

    # Second command
    inner_pipe_data = fetch_data("inner_pipe", inner_diameter_var, inner_schedule_var)
    annulus_data = fetch_data("annulus", annulus_diameter_var, annulus_schedule_var)
    calculate_flow_areas(inner_pipe_data, annulus_data)

    # Calculate fluid velocities
    velocity_annulus, velocity_inner_pipe = calculate_fluid_velocities()

    # Now, call the calc_Re_Nu function
    Re_annulus, Nu_annulus, Re_inner_pipe, Nu_inner_pipe = calc_Re_Nu()

    h_annulus, h_inner_pipe_i, h_inner_pipe_p = calculate_conv_coeff()

    Uo = exchanger_coefficient()

    # Call the calculate_outlet_temps function
    t2_annulus, T2_inner_pipe = calculate_outlet_temps()

    # Calculate heat exchange rate
    q_hx = calculate_hx_rate()

    global t2_annulus_var, T2_inner_pipe_var, Uo_var, heat_transfer_rate_var

    # Now update the GUI StringVar variables with the calculated values
    t2_annulus_var.set(round(t2_annulus, 2))  # Assuming you want to round to 2 decimal places
    T2_inner_pipe_var.set(round(T2_inner_pipe, 2))
    Uo_var.set(round(Uo, 2))
    heat_transfer_rate_var.set(round(q_hx, 2))

   # ... Add more commands as required

def update_flow_image():
    selected_flow = flow_type.get()

    # Delete the existing image on the canvas, if any
    canvas.delete("all")

    if selected_flow == "Counterflow":
        counterflow_image = tk.PhotoImage(file="counterflow.png")
        canvas.create_image(0, 0, anchor=tk.NW, image=counterflow_image)
        canvas.image = counterflow_image
    elif selected_flow == "Parallel Flow":
        parallelflow_image = tk.PhotoImage(file="parallel_flow.png")
        canvas.create_image(0, 0, anchor=tk.NW, image=parallelflow_image)
        canvas.image = parallelflow_image

def create_gui():


    root = tk.Tk()
    root.title("Double-Pipe Heat Exchanger Calculator")
    global fluid_var1, fluid_var2, temp_entry1, temp_entry2, unit_var1, unit_var2, inner_diameter_var, inner_schedule_var, annulus_diameter_var, annulus_schedule_var, mass_flow_rate_entry1, mass_flow_rate_entry2, exchanger_length_entry, flow_type
    global t2_annulus_var, T2_inner_pipe_var, Uo_var, heat_transfer_rate_var, canvas


    #region Create Annulus Frame
    annulus_frame = ttk.LabelFrame(root, text="Annulus (larger pipe, higher velocity)", padding=(10, 5))
    annulus_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

    fluid_label1 = ttk.Label(annulus_frame, text="Fluid Type:")
    fluid_label1.grid(row=0, column=0, padx=10, pady=5)
    fluid_var1 = ttk.Combobox(annulus_frame, values=list(tables.keys()))
    fluid_var1.grid(row=0, column=1, padx=10, pady=5)
    fluid_var1.set("Water")

    temp_label1 = ttk.Label(annulus_frame, text="Inlet Temperature:")
    temp_label1.grid(row=0, column=2, padx=10, pady=5)
    temp_entry1 = ttk.Entry(annulus_frame)
    temp_entry1.grid(row=0, column=3, padx=10, pady=5)

    unit_var1 = ttk.Combobox(annulus_frame, values=["Celsius", "Fahrenheit"])
    unit_var1.grid(row=0, column=4, padx=10, pady=5)
    unit_var1.set("Celsius")

    # Variables for annulus
    annulus_diameter_var = tk.StringVar()
    annulus_schedule_var = tk.StringVar()

    # Nominal Diameter dropdown for annulus
    annulus_diameter_label1 = ttk.Label(annulus_frame, text="Select Nominal Diameter:")
    annulus_diameter_label1.grid(row=1, column=0, padx=10, pady=5)
    annulus_diameter_dropdown = ttk.Combobox(annulus_frame, textvariable=annulus_diameter_var, values=list(pipe_data.keys()))
    annulus_diameter_dropdown.grid(row=1, column=1, padx=5, pady=5)
    annulus_diameter_dropdown.bind('<<ComboboxSelected>>', lambda e: update_schedule_options(annulus_diameter_var, annulus_schedule_dropdown))

    # Schedule dropdown for annulus
    annulus_schedule_label1 = ttk.Label(annulus_frame, text="Select Schedule:")
    annulus_schedule_label1.grid(row=1, column=2, padx=10, pady=5)
    annulus_schedule_dropdown = ttk.Combobox(annulus_frame, textvariable=annulus_schedule_var)
    annulus_schedule_dropdown.grid(row=1, column=3, padx=5, pady=5)

    # Add label for Mass Flow Rate
    mass_flow_rate_label1 = ttk.Label(annulus_frame, text="Mass Flow Rate (kg/s):")
    mass_flow_rate_label1.grid(row=2, column=0, padx=10, pady=5)

    # Add entry box for Mass Flow Rate
    mass_flow_rate_entry1 = ttk.Entry(annulus_frame)
    mass_flow_rate_entry1.grid(row=2, column=1, padx=10, pady=5)

    #endregion

    #region Create Inner Pipe Frame
    inner_pipe_frame = ttk.LabelFrame(root, text="Inner Pipe (smaller pipe)", padding=(10, 5))
    inner_pipe_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

    fluid_label2 = ttk.Label(inner_pipe_frame, text="Fluid Type:")
    fluid_label2.grid(row=0, column=0, padx=10, pady=5)
    fluid_var2 = ttk.Combobox(inner_pipe_frame, values=list(tables.keys()))
    fluid_var2.grid(row=0, column=1, padx=10, pady=5)
    fluid_var2.set("Water")

    temp_label2 = ttk.Label(inner_pipe_frame, text="Inlet Temperature:")
    temp_label2.grid(row=0, column=2, padx=10, pady=5)
    temp_entry2 = ttk.Entry(inner_pipe_frame)
    temp_entry2.grid(row=0, column=3, padx=10, pady=5)

    unit_var2 = ttk.Combobox(inner_pipe_frame, values=["Celsius", "Fahrenheit"])
    unit_var2.grid(row=0, column=4, padx=10, pady=5)
    unit_var2.set("Celsius")

    # Variables for inner pipe
    inner_diameter_var = tk.StringVar()
    inner_schedule_var = tk.StringVar()

    # Diameter dropdown for inner pipe
    inner_diameter_label1 = ttk.Label(inner_pipe_frame, text="Select Nominal Diameter:")
    inner_diameter_label1.grid(row=1, column=0, padx=10, pady=5)
    inner_diameter_dropdown = ttk.Combobox(inner_pipe_frame, textvariable=inner_diameter_var, values=list(pipe_data.keys()))
    inner_diameter_dropdown.grid(row=1, column=1, padx=5, pady=5)
    inner_diameter_dropdown.bind('<<ComboboxSelected>>', lambda e: update_schedule_options(inner_diameter_var, inner_schedule_dropdown))

    # Schedule dropdown for inner pipe
    inner_schedule_label1 = ttk.Label(inner_pipe_frame, text="Select Schedule:")
    inner_schedule_label1.grid(row=1, column=2, padx=10, pady=5)
    inner_schedule_dropdown = ttk.Combobox(inner_pipe_frame, textvariable=inner_schedule_var)
    inner_schedule_dropdown.grid(row=1, column=3, padx=5, pady=5)

    # Add label for Mass Flow Rate
    mass_flow_rate_label2 = ttk.Label(inner_pipe_frame, text="Mass Flow Rate (kg/s):")
    mass_flow_rate_label2.grid(row=2, column=0, padx=10, pady=5)

    # Add entry box for Mass Flow Rate
    mass_flow_rate_entry2 = ttk.Entry(inner_pipe_frame)
    mass_flow_rate_entry2.grid(row=2, column=1, padx=10, pady=5)

    #endregion

    #region Create Exchanger Length Frame
    exchanger_length_frame = ttk.LabelFrame(root, text="Exchanger Length & Flow Direction", padding=(10, 5))
    exchanger_length_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

    # Label for Exchanger Length
    length_label = ttk.Label(exchanger_length_frame, text="Length (m):")
    length_label.grid(row=0, column=0, padx=10, pady=5)

    # Entry box for Exchanger Length
    exchanger_length_entry = ttk.Entry(exchanger_length_frame)
    exchanger_length_entry.grid(row=0, column=1, padx=10, pady=5)

    # Radio Buttons for flow type
    flow_type = tk.StringVar(value="Counterflow")  # default value set to Counterflow

    counterflow_radio = ttk.Radiobutton(exchanger_length_frame, text="Counterflow", variable=flow_type, value="Counterflow", command=update_flow_image)
    counterflow_radio.grid(row=0, column=2, padx=10, pady=5)

    parallelflow_radio = ttk.Radiobutton(exchanger_length_frame, text="Parallel Flow", variable=flow_type, value="Parallel Flow", command=update_flow_image)
    parallelflow_radio.grid(row=0, column=3, padx=10, pady=5)

    # Add a canvas for the image display
    canvas = tk.Canvas(exchanger_length_frame, width=200,height=150)  # Assuming the image dimensions. Adjust width and height accordingly.
    canvas.grid(row=0, column=4, padx=10, pady=5)

    # Call the function once at the start to display the default image
    update_flow_image()
    #endregion

    #region Create Heat Exchanger Results Frame
    results_frame = ttk.LabelFrame(root, text="Heat Exchanger Results", padding=(10, 5))
    results_frame.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

    # Variables to store and display the results
    t2_annulus_var = tk.StringVar(value="N/A")
    T2_inner_pipe_var = tk.StringVar(value="N/A")
    Uo_var = tk.StringVar(value="N/A")
    heat_transfer_rate_var = tk.StringVar(value="N/A")

    # Display Outlet Temperature for Inner Pipe
    ttk.Label(results_frame, text="Inner Pipe Outlet Temp (Celcius):").grid(row=0, column=0, padx=10, pady=5)
    ttk.Label(results_frame, textvariable=T2_inner_pipe_var).grid(row=0, column=1, padx=10, pady=5)

    # Display Outlet Temperature for Annulus
    ttk.Label(results_frame, text="Annulus Outlet Temp (Celcius):").grid(row=1, column=0, padx=10, pady=5)
    ttk.Label(results_frame, textvariable=t2_annulus_var).grid(row=1, column=1, padx=10, pady=5)

    # Display Overall Heat Transfer Coefficient
    ttk.Label(results_frame, text="Overall Heat Transfer Coefficient (Uo):").grid(row=2, column=0, padx=10, pady=5)
    ttk.Label(results_frame, textvariable=Uo_var).grid(row=2, column=1, padx=10, pady=5)

    # Display Heat Transfer Rate
    # Note: Calculation for heat_transfer_rate is not provided in your code.
    # Make sure to calculate it in the `calculate_outlet_temps()` function or in the `on_calculate_click()` function.
    ttk.Label(results_frame, text="Heat Transfer Rate (kW):").grid(row=3, column=0, padx=10, pady=5)
    ttk.Label(results_frame, textvariable=heat_transfer_rate_var).grid(row=3, column=1, padx=10, pady=5)
    #endregion

    #region Buttons
    # Modify the row number for the calculate_btn since we added a new frame above.
    calculate_btn = ttk.Button(root, text="Calculate", command=on_calculate_click)
    calculate_btn.grid(row=3, column=0, columnspan=5, pady=10)
    #endregion

    root.mainloop()

if __name__ == "__main__":
    create_gui()