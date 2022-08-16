from flask import Flask,render_template,request,Markup
import pickle 
import numpy as np
import joblib


l1=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']
test=[' 1. skin can be scraped off and tested for the fungus <br>2.  they may take a scraping of skin cells and have them examined <br>3.  Your doctor can send the swab to a lab, where technicians will culture it to learn what types of fungi or other microbes are present examine these scrapings under a microscope','1.  intradermal test <br>2.  blood test <br>3.  patch test <br>4.  food challenge test',
'1.  upper endoscopy <br>2.  esophageal ph test <br>3.  esophageal manomatery','1. Serum bilirubin test <br>2. Serum albumin test <br>3. Prothrombin time (PTT.  test',
'1. blood test <br>2. skin test','1. Laboratory tests for H. pylori <br>2. Endoscopy <br>3. Upper gastrointestinal series','1.  nucleic acid test (NAT.  2. antigen/antibody test',
'1.  A1C Test <br>2.  Fasting Blood Sugar Test <br>3.  Glucose Tolerance Test <br>4.  Tests for Gestational Diabetes <br>5.  Glucose Screening Test',
'1.  ultrasounds <br>2.  X-rays <br>3.  CT scans','Pulmonary function tests- 1. spirometry  <br>2. exhaled nitric oxide  <br>3. challenge tests','1. Lab tests <br>2. Electrocardiogram <br>3. Echocardiogram <br>4. Ambulatory monitoring',
'1. Magnetic resonance imaging (MRI.  <br>2. Computerized tomography (CT.  scan','1. Neck X-ray <br>2. CT scan <br>3. MRI <br>4. Myelography','1. Electromyogram <br>2. MRI <br>3. CT scan <br>4. X-Rays <br>5. Myelogram checks for spinal cord and nerve injuries',
'1. urinalysis <br>2. blood tests <br>3. imaging tests <br>4. HIDA scan','Artemisinin-based combination therapies','Whole infected cell (wc.  ELISA','1. Dengue NS1 Antigen <br>2. Immunoglobulin M (IgM.  <br>3. Immunoglobulin G (IgG.  <br>4. Dengue RNA PCR test',
'widal test','hepatitis A virus test','hepatitis B blood test','HCV antibody test','blood test','1. anti-HEV IgM <br>2. HEV RNA','1. Liver function tests <br>2. Blood tests <br>3. An ultrasound, CT or MRI scan of the liver <br>4. A liver biopsy',
'1. Blood tests <br>2. Imaging tests <br>3. Sputum tests','No lab tests required to diagnose common colds','1. chest X-ray <br>2. physical exams','1. Digital rectal exam <br>2. Anoscopy <br>3. Sigmoidoscopy','1. Chest X-ray <br>2. Echocardiogram  <br>3. Coronary catheterization (angiogram.  <br>4. Cardiac CT or MRI',
'1. Laser treatment <br>2. Foam sclerotherapy of large veins <br>3. Endoscopic vein surgery <br>4. Sclerotherapy','Blood tests(TSH,T3,T4. ','1. Blood tests <br>2. Radioiodine uptake test <br>3. Thyroid scan <br>4. Thyroid ultrasound',' hemoglobin A1c','1. Imaging tests <br>2. Blood tests <br>3. Joint fluid analysis',
'1. Electronystagmography (ENG.  <br>2. videonystagmography (VNG.  <br>3. Magnetic resonance imaging (MRI. ','1.  Dehydroepiandrosterone sulphate (DHEA-S.  <br>2.  Lipid profile test ','1. Analyzing a urine sample <br>2. Growing urinary tract bacteria in a lab <br>3. Creating images of your urinary tract',
'may use small sample of skin to test under microscope','your doctor might take a sample of the liquid produced by a sore and test it to see what types of antibiotics would work best on it']
precautions=['1. keep your skin clean and dry, particularly the folds of your skin <br>2. wash your hands often, especially after touching animals or other people <br>3. avoid using other peoples towels and other personal care products', '1. Avoid your allergens<br>2. Take your medicines as prescribed<br>3. If you are at risk for anaphylaxis, keep your epinephrine auto-injectors with you at all times<br>4. Wear a medical alert bracelet (or necklace)', '1. Lose weight <br>2. Avoid foods known to cause reflux <br>3. Eat smaller meals <br>4. Quit smoking', '1. sugars and highly refined foods, such as white bread and corn syrup<br>2. soy products<br>3. processed meats<br>4. full-fat dairy produce', '1. Limit alcohol <br>2. Talk to your pharmacist <br>3. Know how to take the drug <br>4. Be suspicious of supplements', '1. Avoid tobacco products <br>2. Use caution with aspirin and/or NSAIDs <br>3. Dont ignore your ulcer symptoms', '1. abstinenceB (not having sex.  <br>2. never sharing needles <br>3. using condoms', '1. Cut Sugar and Refined Carbs From Your Diet <br>2. Work Out Regularly <br>3. Drink Water as Your Primary Beverage', 'wearing gloves and a plastic apron or impervious gownB when having contact with the patient or the patients environment, especially when attending to patient toileting and hygiene Protective eyewear and mask must be worn when there is the potential of vomit or faecal splashing', '1. Stay Away From Allergens <br>2. Avoid Smoke of Any Type <br>3. Prevent Colds <br>4. Get Your Vaccinations <br>5. Allergy-Proof Your Home', '1. Eat a Healthy Diet <br>2. Be Physically Active <br>3. Get Enough Sleep <br>4. Keep Yourself at a Healthy Weight', 'Eat regularly. Dont skip meals Curb the caffeine Be careful with exercise Get regular shut-eye', '1. Stay physically active <br>2. Use good posture <br>. Avoid trauma to your neck <br>4. Prevent neck injuries by always using the right equipment and the  right form when exercising or playing sports', '1. Lose weight <br>2. Exercise more <br>3. Treat atrial fibrillation <br>4. If you drink booz do it in moderation <br>5. Treat diabetes', 'Drink at least eight glasses of fluids per day Consider adding milk thistle to your routine Opt for fruits Look for high-fiber foods', 'Drape mosquito netting over beds Put screens on windows and doors Wear long pants and long sleeves to cover your skin Apply mosquito repellent with DEET (diethyltoluamide.  to exposed skin', 'get the chickenpox vaccine', 'Apply mosquito repellent, ideally one containing DEET Wear long-sleeves and long pants to cover your arms and legs Use mosquito nets while sleeping', 'Wash your hands Avoid drinking untreated water Avoid raw fruits and vegetables Choose hot foods', '1. Avoid raw shellfish<br>2. Beware of sliced fruit that may have been washed in contaminated water<br>3. do not buy food from street vendors', '1. Know the HBV status of your sexual partner<br>2. Use a new latex or polyurethane condom every time you have sex<br>3. Dont use illegal drugs', '1. Never share needles<br>2. Avoid direct exposure to blood or blood products<br>3. Dont share personal care items<br>4. Choose tattoo and piercing parlors carefully', '1. Avoid sharing drug equipment<br>2. Practice safe sex<br>3. Wear latex gloves if you are likely to be in contact with someone else blood or bodily fluids<br>4. Avoid dental, medical or cosmetic procedures that penetrate the skin with unsterilized equipment', '1. keep good sanitization<br>2. drink clean and hygiene water', 'stop drinking alcohol', 'Wash your hands after coughing or sneezing Always cover your mouth with a tissue when you cough or sneeze Take all of your medicines as they are prescribed, until your doctor takes you off them', 'Wash your hands Disinfect your stuff Cover your cough', '1.  Dont smoke <br>2.  Get vaccinated <br>3.  Practice good hygiene <br>4.  Keep your immune system strong', 'Dont sit too long or push too hard on the toilet Drink plenty of water Stay physically active', '1.  Stop smoking <br>2.  Choose good nutrition <br>3.  Lower high blood pressure <br>4.  Be physically active every day <br>5.  Reduce stress', '1. Exercising <br>2. Watching your weight <br>3. Eating a high-fiber <br>4. Elevating your legs', '1. excercise daily <br>2. take good diet 3. reduce weight', 'Avoid other foods high in iodine', '1. Monitor your blood sugar <br>2. Dont skip or delay meals or snacks <br>3. Measure medication carefully, and take it on time', '1. Keep a healthy body weight <br>2. Control your blood sugar <br>3. Pay attention to pain <br>4. Prevent injury to your joints', '1. Avoid driving <br>2. Avoid working at heights <br>3. Wear shoes with low heels and nonslip soles <br>4. Keep your shoes tied', '1. Wash the face twice daily <br>2. Refrain from harsh scrubbing <br>3. Keep hair clean <br>4. Apply topical treatments', '1. Drink plenty of liquids, especially water <br>2. Drink cranberry juice <br>3. Wipe from front to back', '1. Use Moisturizing Lotions <br>2. Use a Humidifier <br>3. Use Moisturizing Lotions', '1. Wash your hands with soap <br>2. use alcohol hand rubs <br>3. Do not share personal items']

    
l2=[]
for x in range(0,len(l1)):
  l2.append(0)
disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
        'Peptic ulcer disease','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
        ' Migraine','Cervical spondylosis',
        'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']
app = Flask(__name__)
app._static_folder = "/Users/lakshya_mangal/Desktop/minorproject_1/static"
model =pickle.load(open('model.pkl','rb'))
@app.route("/")
def hello_world():
    return render_template('prediction.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
 gnb=joblib.load('model.pkl')
 psymptoms=[x for x in request.form.values()]
#  a=len(psymptoms)
#  print(psymptoms)
#  return render_template('prediction.html')

 for k in range(0,len(l1)):
   for z in psymptoms:
              if(z==l1[k]):
                  l2[k]=1

 inputtest = [l2]
 predict = gnb.predict(inputtest)
 predicted=predict[0]
 h='no'
 for a in range(0,len(disease)):
          if(disease[predicted] == disease[a]):
              h='yes'
              break
 b=len(psymptoms)
 if (h=='yes' and b>3):
      return render_template('second.html',pred=disease[a],prec=Markup(precautions[a]),tes=Markup(test[a]))
      # return jsonify({"disease":disease[a]})
 else:
    return render_template('prediction.html',pred='NO DISEASE FOUND')
    
   
