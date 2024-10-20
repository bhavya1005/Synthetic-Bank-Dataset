import random
from faker import Faker
import numpy as np
import csv
import os

# Initialize Faker library for generating fake data
fake = Faker()
Faker.seed(42)  # Seed for reproducibility

# List of U.S. state abbreviations
us_states = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

# Assign random weights to all states
np.random.seed(42)  # Set seed for reproducibility (optional)
random_weights = np.random.rand(len(us_states))  # Generate random weights
state_probabilities = random_weights / random_weights.sum()  # Normalize weights to sum to 1

# Create a dictionary mapping states to their probabilities
state_weights = dict(zip(us_states, state_probabilities))

# List of common American last names
common_last_names = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
    'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
    'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
    'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
    'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell',
    'Carter', 'Roberts'
]

# List of common addresses (approximately 50 addresses)
common_addresses = [
    '123 Maple St, Springfield, IL, 62701',
    '456 Oak Ave, Madison, WI, 53703',
    '789 Pine Rd, Austin, TX, 73301',
    '101 Birch Blvd, Denver, CO, 80203',
    '202 Elm St, Seattle, WA, 98101',
    '303 Cedar St, Boston, MA, 02101',
    '404 Cherry Ln, Columbus, OH, 43215',
    '505 Walnut St, Nashville, TN, 37203',
    '606 Ash Dr, Phoenix, AZ, 85001',
    '707 Poplar Ct, Portland, OR, 97201',
    '808 Magnolia Way, Atlanta, GA, 30303',
    '909 Cypress Cir, Miami, FL, 33101',
    '1001 Willow Rd, San Francisco, CA, 94101',
    '1102 Redwood Dr, San Diego, CA, 92101',
    '1203 Aspen Pl, Dallas, TX, 75201',
    '1304 Hickory Ln, Charlotte, NC, 28202',
    '1405 Dogwood St, Kansas City, MO, 64101',
    '1506 Alder Ave, Indianapolis, IN, 46201',
    '1607 Beech Blvd, Jacksonville, FL, 32099',
    '1708 Sycamore Rd, Chicago, IL, 60601',
    '1809 Spruce St, Philadelphia, PA, 19102',
    '1901 Fir St, Houston, TX, 77002',
    '2002 Pineapple Dr, Honolulu, HI, 96801',
    '2103 Palm Blvd, Los Angeles, CA, 90001',
    '2204 Oakwood Rd, Detroit, MI, 48201',
    '2305 Maplewood Ave, Minneapolis, MN, 55401',
    '2406 Lakeview Dr, Cleveland, OH, 44101',
    '2507 River Rd, New Orleans, LA, 70112',
    '2608 Mountain Dr, Denver, CO, 80202',
    '2709 Valley Rd, Salt Lake City, UT, 84101',
    '2801 Ocean Ave, Virginia Beach, VA, 23450',
    '2902 Desert Rd, Albuquerque, NM, 87101',
    '3003 Prairie St, Omaha, NE, 68102',
    '3104 Meadow Ln, Lexington, KY, 40507',
    '3205 Forest Ave, Des Moines, IA, 50309',
    '3306 Park Blvd, San Jose, CA, 95101',
    '3407 Riverbend Rd, Tulsa, OK, 74103',
    '3508 Hilltop Dr, Birmingham, AL, 35203',
    '3609 Sunrise St, Tampa, FL, 33602',
    '3701 Sunset Blvd, Los Angeles, CA, 90028',
    '3802 Moonlight Ln, Orlando, FL, 32801',
    '3903 Starlight Rd, Las Vegas, NV, 89101',
    '4004 Galaxy Ave, Houston, TX, 77003',
    '4105 Comet St, Phoenix, AZ, 85002',
    '4206 Meteor Dr, Dallas, TX, 75202',
    '4307 Planet Pl, Austin, TX, 73301',
    '4408 Rocket Rd, Cape Canaveral, FL, 32920',
    '4509 Satellite St, Huntsville, AL, 35801',
    '4601 Nebula Ct, Pasadena, CA, 91101',
    '4702 Eclipse Ave, Portland, OR, 97202',
    '4803 Equinox Rd, Seattle, WA, 98102',
    '4904 Solstice St, Anchorage, AK, 99501',
    '5005 Aurora Ln, Fairbanks, AK, 99701'
]

# Function to generate and save records to a CSV file
def generate_records_to_csv():
    # Define the CSV file path
    csv_file_path = 'customers.csv'

    # Define the header for the CSV
    header = [
        'First_Name', 'Last_Name', 'Address', 'State', 'Pincode',
        'Income', 'Phone_Number', 'Role', 'Credit_Score', 'Age', 'Created_At'
    ]

    # Total number of records to generate
    total_records = 100000  # Adjust this number as needed

    # Number of records with specified last names and addresses
    portion_with_common_data = 20000  # Adjust as needed
    # Number of records with random data
    portion_with_random_data = total_records - portion_with_common_data

    # Define missingness rates per column (20% missingness)
    missingness_rates = {
        'Income': 0.2,
        'Credit_Score': 0.2,
        'Age': 0.2
    }

    # List of states and their probabilities
    states = list(state_weights.keys())
    probabilities = list(state_weights.values())

    # Function to randomly introduce missing values
    def maybe_null(column):
        return '' if random.random() < missingness_rates.get(column, 0) else None

    # Batch size for writing records
    batch_size = 1000
    records = []

    # Ensure the CSV file is created fresh
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    try:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header
            writer.writerow(header)

            # Function to generate a single record
            def generate_record(common_data=True):
                if common_data:
                    # Generate records with specified last names and addresses
                    first_name = fake.first_name()
                    last_name = random.choice(common_last_names)
                    address = random.choice(common_addresses)

                    # Extract state and pincode from the address
                    address_parts = address.split(', ')
                    if len(address_parts) >= 4:
                        state = address_parts[-2]
                        pincode = address_parts[-1]
                    else:
                        # If parsing fails, assign random state and pincode
                        state = np.random.choice(states, p=probabilities)
                        pincode = fake.postcode()

                else:
                    # Generate records with random data
                    first_name = fake.first_name()
                    last_name = fake.last_name()
                    street_address = fake.street_address()
                    city = fake.city()
                    state = np.random.choice(states, p=probabilities)
                    
                    # Ensure state abbreviation is correctly used
                    # Faker's zipcode_in_state may expect full state name, so handle accordingly
                    # Alternatively, generate a random zipcode
                    pincode = fake.zipcode()

                    # Construct full address
                    address = f"{street_address}, {city}, {state}, {pincode}"

                # Generate realistic age
                age = int(np.random.normal(40, 12))  # Mean age 40, std dev 12
                age = max(18, min(age, 90))  # Ensure age is between 18 and 90

                # Generate realistic income based on age
                if age < 25:
                    mean_income = 30000
                    sigma_income = 0.6
                elif age < 35:
                    mean_income = 50000
                    sigma_income = 0.5
                elif age < 50:
                    mean_income = 75000
                    sigma_income = 0.4
                elif age < 65:
                    mean_income = 60000
                    sigma_income = 0.5
                else:
                    mean_income = 40000
                    sigma_income = 0.6
                income = np.random.lognormal(mean=np.log(mean_income), sigma=sigma_income)
                income = round(min(income, 200000), 2)  # Cap income at 200,000

                # Generate other details
                phone_number = fake.phone_number()
                # Standardize phone number format (e.g., (XXX) XXX-XXXX)
                phone_number = fake.phone_number()
                role = random.choices(['Primary', 'Secondary'], weights=[0.7, 0.3])[0]
                
                # Generate credit score somewhat correlated with income
                base_credit = 600
                income_factor = (income / 100000) * 100  # Scale income to influence credit score
                credit_score = int(min(850, max(300, np.random.normal(base_credit + income_factor, 50))))

                created_at = fake.date_time_this_year().strftime('%Y-%m-%d %H:%M:%S')

                # Introduce missing values where applicable
                income = maybe_null('Income') if random.random() < missingness_rates['Income'] else income
                credit_score = maybe_null('Credit_Score') if random.random() < missingness_rates['Credit_Score'] else credit_score
                age = maybe_null('Age') if random.random() < missingness_rates['Age'] else age

                # Return the record as a tuple
                return (
                    first_name, last_name, address, state, pincode, income,
                    phone_number, role, credit_score, age, created_at
                )

            # Generate records with specified last names and addresses
            for _ in range(portion_with_common_data):
                record = generate_record(common_data=True)
                records.append(record)

                # Write batch to CSV
                if len(records) >= batch_size:
                    writer.writerows(records)
                    records = []

            # Generate records with random data
            for _ in range(portion_with_random_data):
                record = generate_record(common_data=False)
                records.append(record)

                # Write batch to CSV
                if len(records) >= batch_size:
                    writer.writerows(records)
                    records = []

            # Write any remaining records
            if records:
                writer.writerows(records)

        print(f"Data successfully written to {csv_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function to generate and save records when the script is executed
if __name__ == "__main__":
    generate_records_to_csv()
