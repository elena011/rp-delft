import numpy as np

def transform_age(age, adult_age=20):
    age = np.asarray(age)
    transformed = np.where(
        age <= adult_age,
        np.log(age + 1) - np.log(adult_age + 1),
        (age - adult_age) / (adult_age + 1)
    )
    return transformed

def anti_transform_age(transformed_age, adult_age=20):
    transformed_age = np.asarray(transformed_age)
    age = np.where(
        transformed_age <= 0,
        np.exp(transformed_age + np.log(adult_age + 1)) - 1,
        (adult_age + 1) * transformed_age + adult_age
    )
    return age
