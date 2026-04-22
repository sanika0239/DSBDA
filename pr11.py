def calculate_averages(filename):
    total_temp = 0
    total_dew = 0
    total_wind = 0
    count = 0

    with open(filename, 'r') as file:
        next(file)  # Skip header

        for line in file:
            parts = line.split()
            
            temp = float(parts[1])
            dew = float(parts[2])
            wind = float(parts[3])

            total_temp += temp
            total_dew += dew
            total_wind += wind
            count += 1

    avg_temp = total_temp / count
    avg_dew = total_dew / count
    avg_wind = total_wind / count

    print("Average Temperature:", avg_temp)
    print("Average Dew Point:", avg_dew)
    print("Average Wind Speed:", avg_wind)


# Run the program
calculate_averages("sample_weather.txt")