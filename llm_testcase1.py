
from langchain_openai import OpenAI

# We are importing secreat key from  f secretkey.py file which is stored in our folder to make confidentiality
# Hi  hhhh

from constant import secreat_key

llm_prompt=OpenAI(openai_api_key=secreat_key)

from langchain.prompts import PromptTemplate

# prompt_template = PromptTemplate(
#     input_variables=["product"],
#     template="Given a {product} with a single login button. Write test scenarios about the {product}?",
# )

# response=llm_prompt.invoke("Web application")
# print(response)

from langchain.prompts.few_shot import FewShotPromptTemplate

examples = [
    {"Usecase": "I need test cases to open account in ICICI bank",

"Test cases": """Test Case 1: Basic Account Opening

Test Case Description: Attempt to open a savings account with valid required information.

Inputs:
Name: John Doe
Date of Birth: 01-01-1990
Address: [Provide valid address]
Identification (ID) Proof: Aadhar Card
Initial Deposit: Rs. 10,000

Expected Outcome: Account successfully opened with the provided details.

Test Case 2: Invalid Identification Proof

Test Case Description: Attempt to open an account with an invalid or expired ID proof.

Inputs:
Name: Jane Smith
Date of Birth: 15-05-1985
Address: [Provide valid address]
Identification (ID) Proof: Expired Passport
Initial Deposit: Rs. 5,000

Expected Outcome: Account opening process should fail due to invalid ID proof.

Test Case 3: Missing Information

Test Case Description: Attempt to open an account with missing mandatory information.

Inputs:
Name: Mark Johnson
Date of Birth: 10-12-1982
Address: [Provide valid address]
Identification (ID) Proof: Voter ID Card
Initial Deposit: Rs. 20,000

Expected Outcome: Account opening process should fail due to missing required details (e.g., missing phone number or incomplete address).

Test Case 4: Minimum Initial Deposit

Test Case Description: Attempt to open an account with the minimum required initial deposit.

Inputs:
Name: Sarah Lee
Date of Birth: 25-09-1995
Address: [Provide valid address]
Identification (ID) Proof: Driving License
Initial Deposit: Rs. 1,000

Expected Outcome: Account opening should succeed with the minimum deposit amount.

Test Case 5: Large Initial Deposit

Test Case Description: Attempt to open an account with a large initial deposit.

Inputs:
Name: Robert Brown
Date of Birth: 03-04-1980
Address: [Provide valid address]
Identification (ID) Proof: PAN Card
Initial Deposit: Rs. 1,000,000

Expected Outcome: Account opening should succeed with the large deposit, if within bank limits.

Test Case 6: Duplicate Account Opening

Test Case Description: Attempt to open an account with the same details as an existing account holder.

Inputs:
Name: John Doe (same as existing account holder)
Date of Birth: 01-01-1990
Address: [Provide valid address]
Identification (ID) Proof: Aadhar Card
Initial Deposit: Rs. 15,000

Expected Outcome: Account opening process should fail due to duplicate account details.

Test Case 7: Error Handling

Test Case Description: Attempt to open an account with server/connection errors.

Inputs:
Name: Emily White
Date of Birth: 20-07-1988
Address: [Provide valid address]
Identification (ID) Proof: Passport
Initial Deposit: Rs. 8,000

Expected Outcome: Account opening process should gracefully handle any server or connection errors and provide appropriate feedback.

"""},

{"Usecase": "I need a testcases to open gmail account",

 "Test cases": """ Test Case 1: Successful Account Creation

Test Case Description: Attempt to create a new Gmail account with valid information.

Test Steps: a. Open the Gmail signup page. b. Enter valid details in the required fields:
Name: John Doe
Username: johndoe123 (unique and available)
Password: [Secure password]
Date of Birth: 01-01-1990
Gender: Male
Mobile Number: [Valid phone number] c. Complete any additional verification steps (if required). d. Accept the terms and conditions. e. Click on "Next" to create the account.

Expected Outcome: A new Gmail account is successfully created, and the user is logged into their new inbox.

Test Case 2: Invalid Username

Test Case Description: Attempt to create a Gmail account with an invalid or already taken username.

Test Steps: a. Open the Gmail signup page. b. Enter invalid or existing username in the username field. c. Enter valid details in other required fields. d. Complete any additional verification steps (if required). e. Click on "Next" to create the account.

Expected Outcome: Account creation process fails with an error message indicating that the username is invalid or already taken.

Test Case 3: Weak Password

Test Case Description: Attempt to create a Gmail account with a weak password.
Test Steps: a. Open the Gmail signup page. b. Enter valid details including a weak password (e.g., "password" or "123456"). c. Complete any additional verification steps (if required). d. Click on "Next" to create the account.

Expected Outcome: Account creation process fails with an error message indicating that the password is too weak.

Test Case 4: Missing Information

Test Case Description: Attempt to create a Gmail account with missing mandatory information.

Test Steps: a. Open the Gmail signup page. b. Leave one or more required fields (e.g., Name, Date of Birth) empty. c. Complete other fields with valid information. d. Complete any additional verification steps (if required). e. Click on "Next" to create the account.

Expected Outcome: Account creation process fails with an error message indicating the missing required information.
Test Case 5: Age Restriction

Test Case Description: Attempt to create a Gmail account for a user below the minimum age requirement.

Test Steps: a. Open the Gmail signup page. b. Enter valid details including a date of birth for a user under 13 years old. c. Complete any additional verification steps (if required). d. Click on "Next" to create the account.

Expected Outcome: Account creation process fails with an error message indicating that the user does not meet the minimum age requirement.

Test Case 6: Account Recovery Setup

Test Case Description: Verify the ability to set up account recovery options during account creation.

Test Steps: a. Open the Gmail signup page. b. Enter valid details including a recovery email address and/or phone number. c. Complete any additional verification steps (if required). d. Click on "Next" to create the account.

Expected Outcome: Account is successfully created with account recovery options set up for security purposes.

Test Case 7: Confirmation Email

Test Case Description: Verify the receipt of a confirmation email after successful account creation.

Test Steps: a. Open the Gmail inbox of the newly created account. b. Check for the presence of a confirmation email from Google. c. Click on the confirmation link/button within the email (if required).

Expected Outcome: Confirmation email is received and the account is fully activated for use.
"""}


           ]


example_prompt = PromptTemplate(input_variables=["Usecase", "Test cases"], template="Usecase: {Usecase}\n{Test cases}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples, 
    example_prompt=example_prompt, 
    suffix="Usecase: {input}", 
    input_variables=["input"]
)


# response=llm_prompt.invoke(myprompt)
# print(response)

import streamlit as st

st.title("Testcase Generation Demo with Help of LLM and Langchain")

input_text=st.text_input("Search the usecases which we required")

myprompt=few_shot_prompt.format(input=input_text)

if input_text:
    st.write(llm_prompt.invoke(myprompt))