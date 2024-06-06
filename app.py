
from flask import Flask, render_template, request

# Import Module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def predict_fraud_with_accuracy(input_data):

    dataOrigin = pd.read_excel ("[Dataset]_(Asuransi).xlsx",sheet_name="in")
    
    # Bersihkan kolom yang terlalu redudant

    data = dataOrigin
    # Membersihkan sex
    data['insured sex'] = data['insured_sex_MALE'].apply(lambda x: 'Male' if x else 'Female')
    data = data.drop(columns=['insured_sex_MALE', 'insured_sex_FEMALE'])
    data.head()

    # Membersihkan occupantion
    def get_occupation(row):
        for col in occupation_columns:
            if row[col] == 1:
                return col.replace('insured_occupation_', '')
        return 'Unknown'

    occupation_columns = [
        'insured_occupation_adm-clerical', 'insured_occupation_armed-forces',
        'insured_occupation_craft-repair', 'insured_occupation_exec-managerial',
        'insured_occupation_farming-fishing', 'insured_occupation_handlers-cleaners',
        'insured_occupation_machine-op-inspct', 'insured_occupation_other-service',
        'insured_occupation_priv-house-serv', 'insured_occupation_prof-specialty',
        'insured_occupation_protective-serv', 'insured_occupation_sales',
        'insured_occupation_tech-support', 'insured_occupation_transport-moving'
    ]

    data['occupation'] = data.apply(get_occupation, axis=1)
    data = data.drop(columns=occupation_columns)

    # Membersihkan hobbies
    hobbies_columns = [
        'insured_hobbies_chess', 'insured_hobbies_cross-fit', 'insured_hobbies_other'
    ]

    def get_hobbies(row):
        for col in hobbies_columns:
            if row[col] == 1:
                return col.replace('insured_hobbies_', '')
        return 'Unknown'

    data['hobbies'] = data.apply(get_hobbies, axis=1)
    data = data.drop(columns=hobbies_columns)

    # Membersihkan incident type
    incidentType_columns = [
        'incident_type_Multi-vehicle Collision', 'incident_type_Parked Car',
        'incident_type_Single Vehicle Collision', 'incident_type_Vehicle Theft'
    ]

    def get_incidentType(row):
        for col in incidentType_columns:
            if row[col] == 1:
                return col.replace('incident_type_', '')
        return 'Unknown'

    data['incident type'] = data.apply(get_incidentType, axis=1)
    data = data.drop(columns=incidentType_columns)

    # Membersihkan collision type
    collisionType_columns = [
        'collision_type_?', 'collision_type_Front Collision',
        'collision_type_Rear Collision', 'collision_type_Side Collision'
    ]

    def get_collisionType(row):
        for col in collisionType_columns:
            if row[col] == 1:
                return col.replace('collision_type_', '')
        return 'Unknown'

    data['collision type'] = data.apply(get_collisionType, axis=1)
    data = data.drop(columns=collisionType_columns)

    # Membersihkan incident severity
    incidentSeverity_columns = [
        'incident_severity_Major Damage', 'incident_severity_Minor Damage',
        'incident_severity_Total Loss', 'incident_severity_Trivial Damage'
    ]

    def get_incidentSeverity(row):
        for col in incidentSeverity_columns:
            if row[col] == 1:
                return col.replace('incident_severity_', '')
        return 'Unknown'

    data['incident severity'] = data.apply(get_incidentSeverity, axis=1)
    data = data.drop(columns=incidentSeverity_columns)

    # Membersihkan autoruties contacted
    authoritiesContacted_columns = [
        'authorities_contacted_Ambulance', 'authorities_contacted_Fire',
        'authorities_contacted_None', 'authorities_contacted_Other',
        'authorities_contacted_Police'
    ]

    def get_authoritiesContacted(row):
        for col in authoritiesContacted_columns:
            if row[col] == 1:
                return col.replace('authorities_contacted_', '')
        return 'Unknown'

    data['authorities contacted'] = data.apply(get_authoritiesContacted, axis=1)
    data = data.drop(columns=authoritiesContacted_columns)

    # Membersihkan age group
    ageGroup_columns = [
    'age_group_15-20', 'age_group_21-25',
    'age_group_26-30', 'age_group_31-35',
    'age_group_36-40', 'age_group_41-45',
    'age_group_46-50', 'age_group_51-55',
    'age_group_56-60', 'age_group_61-65'
    ]

    def get_ageGroup(row):
        for col in ageGroup_columns:
            if row[col] == 1:
                return col.replace('age_group_', '')
        return 'Unknown'

    data['age group'] = data.apply(get_ageGroup, axis=1)
    data = data.drop(columns=ageGroup_columns)

    # Membersihkan month as customer group
    monthAsCustomerGroup_columns = [
    'months_as_customer_groups_0-50', 'months_as_customer_groups_51-100',
    'months_as_customer_groups_101-150', 'months_as_customer_groups_151-200',
    'months_as_customer_groups_201-250', 'months_as_customer_groups_251-300',
    'months_as_customer_groups_301-350', 'months_as_customer_groups_351-400',
    'months_as_customer_groups_401-450', 'months_as_customer_groups_451-500'
    ]

    def get_monthAsCustomerGroup(row):
        for col in monthAsCustomerGroup_columns:
            if row[col] == 1:
                return col.replace('months_as_customer_groups_', '')
        return 'Unknown'

    data['months as customer groups'] = data.apply(get_monthAsCustomerGroup, axis=1)
    data = data.drop(columns=monthAsCustomerGroup_columns)

    # Membersihkan policy anual premium group
    policyAnualPremiumGroup_columns = [
        'policy_annual_premium_groups_high', 'policy_annual_premium_groups_low',
        'policy_annual_premium_groups_medium', 'policy_annual_premium_groups_very high',
        'policy_annual_premium_groups_very low'
    ]

    def get_policyAnualPremiumGroup(row):
        for col in policyAnualPremiumGroup_columns:
            if row[col] == 1:
                return col.replace('policy_annual_premium_groups_', '')
        return 'Unknown'

    data['policy anual premium group'] = data.apply(get_policyAnualPremiumGroup, axis=1)
    data = data.drop(columns=policyAnualPremiumGroup_columns)

    data = data.drop(columns=['Unnamed: 0'])

    # Pembuatan model prediksi
    dataProcess = data

    # Pembuatan matriks berdasarkan data
    X = dataProcess.drop(columns=['fraud_reported'])
    y = dataProcess['fraud_reported']

    # Memisahkan kolom dengan kategori dan integer
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns


    numerical_transformer = StandardScaler()


    categorical_transformer = OneHotEncoder(handle_unknown='ignore')


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])


    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_pipeline.fit(X_train, y_train)

    joblib.dump(model_pipeline, 'insurance_fraud_model.pkl')

    # Merubah data menjadi data frame
    sample_df = pd.DataFrame(input_data)

    prediction = model_pipeline.predict(sample_df)

    y_pred_test = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    return prediction, accuracy

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    if request.method == 'POST':
            capitalGain = request.form.get('capitalGain')
            capitalLoss = request.form.get('capitalLoss')
            incidentHour = request.form.get('incidentHour')
            vehicleCount = request.form.get('vehicleCount')
            witnessesCount = request.form.get('witnessesCount')
            claimAmount = request.form.get('claimAmount')
            gender = request.form.get('gender')
            occupation = request.form.get('occupation')
            hobby = request.form.get('hobby')
            ageGroup = request.form.get('ageGroup')
            incidentType = request.form.get('incidentType')
            collisionType = request.form.get('collisionType')
            incidentSeverity = request.form.get('incidentSeverity')
            authoritiesContacted = request.form.get('authoritiesContacted')
            MACG = request.form.get('MACG')
            PAPG = request.form.get('PAPG')
            output = "<hr>"

            # Contoh Skenario
            inputData= {
                'capital-gains': [capitalGain],
                'capital-loss': [capitalLoss],
                'incident_hour_of_the_day': [incidentHour],
                'number_of_vehicles_involved': [vehicleCount],
                'witnesses': [witnessesCount],
                'total_claim_amount': [claimAmount],
                'insured sex': [gender],
                'occupation': [occupation],
                'hobbies': [hobby],
                'incident type': [ageGroup],
                'collision type': [incidentType],
                'incident severity': [collisionType],
                'authorities contacted': [incidentSeverity],
                'age group': [authoritiesContacted],
                'months as customer groups': [MACG],
                'policy anual premium group': [PAPG]
            }

            # Memprediksi berdasarkan data apakah sebuah penipuan asuransi atau bukan & akurasinya
            is_fraud, model_accuracy = predict_fraud_with_accuracy(inputData)
            if is_fraud:
                output += "Penipuan Asuransi"
            else:
                output += "Bukan Penipuan Asuransi"
            output += "<br>"
            output += "Akurasi Model = "
            output += str(model_accuracy)          
            

    # Render the form in the HTML template
    return render_template('template.html',
                        output1=output)

if __name__ == '__main__':
    app.run(debug=True)