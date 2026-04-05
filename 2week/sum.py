import pandas as pd

class Data:
    def __init__(self, path):
        self.path = path
        # Read CSV and skip problematic lines
        try:
            self.data = pd.read_csv(path, encoding='latin-1', skiprows=52, usecols=[0, 1], 
                                   on_bad_lines='skip')
        except:
            # Fallback: read without dtype specification
            self.data = pd.read_csv(path, encoding='latin-1', skiprows=52, usecols=[0, 1])
        
        # Rename columns for clarity
        self.data.columns = ['wavelength', 'irradiance']
        
        # Convert to numeric, coercing errors to NaN
        self.data['wavelength'] = pd.to_numeric(self.data['wavelength'], errors='coerce')
        self.data['irradiance'] = pd.to_numeric(self.data['irradiance'], errors='coerce')
        
    def sum_spectral_data(self):
        """Sum the spectral data from wavelengths 380-780nm"""
        # Filter data between 380-780nm and remove NaN values
        spectral_data = self.data[(self.data['wavelength'] >= 380) & (self.data['wavelength'] <= 780)]
        spectral_data = spectral_data.dropna()
        # Sum the irradiance values and convert to regular Python float
        total_sum = float(spectral_data['irradiance'].sum())
        return total_sum

'''
# Example usage
if __name__ == "__main__":
    # Read and process each CSV file
    import os
    data_dir = "data"
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            data = Data(filepath)
            total = data.sum_spectral_data()
            print(f"{filename}: {total:.2f}")
        
'''