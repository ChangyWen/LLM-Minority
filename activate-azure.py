import requests
import uuid
import datetime
from azure.identity import AzureCliCredential, get_bearer_token_provider


credential = AzureCliCredential()

token_provider = get_bearer_token_provider(
    credential, "https://management.azure.com/.default"
)


def get_role_assignments():
    url = f"https://management.azure.com/providers/Microsoft.Authorization/roleEligibilitySchedules?api-version=2020-10-01&%24filter=asTarget()"

    headers = {
        "Authorization": "Bearer " + token_provider(),
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("List role assignments request submitted successfully.")
        return response.json()
    else:
        print("Failed to list role assignments request.")
        print(response.text)


def get_role_instance():
    url = f"https://management.azure.com/providers/Microsoft.Authorization/roleEligibilityScheduleInstances?api-version=2020-10-01&%24filter=asTarget()"

    headers = {
        "Authorization": "Bearer " + token_provider(),
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200 or response.status_code == 201:
        print("Get role instance request submitted successfully.")
        return response.json()
    else:
        print("Failed to get role instance request.")
        print(response.text)


def activate_role_assignment(
    scope,
    role_definition_id,
    principal_id,
    role_eligibility_schedule_id,
    role_assignment_schedule_id,
):
    url = f"https://management.azure.com{scope}/providers/Microsoft.Authorization/roleAssignmentScheduleRequests/{role_assignment_schedule_id}?api-version=2020-10-01"

    start_time = datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    data = {
        "Properties": {
            "RoleDefinitionId": role_definition_id,
            "PrincipalId": f"{principal_id}",
            "RequestType": "SelfActivate",
            "Justification": "Access",
            "ScheduleInfo": {
                "StartDateTime": None,
                "Expiration": {"Type": "AfterDuration", "Duration": "PT480M"},
            },
            "LinkedRoleEligibilityScheduleId": role_eligibility_schedule_id,
        }
    }

    headers = {
        "Authorization": "Bearer " + token_provider(),
        "Content-Type": "application/json",
    }

    response = requests.put(url, json=data, headers=headers)

    if response.status_code == 200 or response.status_code == 201:
        print("Role activation request submitted successfully.")
    else:
        print("Failed to submit role activation request.")
        print(response.status_code, response.text)


rslt = get_role_instance()

my_object_id = "90bb4e23-2e90-4ec9-b6a1-772a0ce79827"

for role_assignment in rslt["value"]:
    scope = role_assignment["properties"]["scope"]
    role_definition_id = role_assignment["properties"]["roleDefinitionId"]
    principal_id = role_assignment["properties"]["principalId"]
    role_eligibility_schedule_id = role_assignment["properties"][
        "roleEligibilityScheduleId"
    ]
    role_assignment_schedule_id = uuid.uuid4()

    print(
        "Principal: ",
        role_assignment["properties"]["expandedProperties"]["principal"]["displayName"],
    )

    print(
        "Role Definition: ",
        role_assignment["properties"]["expandedProperties"]["roleDefinition"][
            "displayName"
        ],
    )

    print(
        "Scope: ",
        role_assignment["properties"]["expandedProperties"]["scope"]["displayName"],
    )

    activate_role_assignment(
        scope,
        role_definition_id,
        my_object_id,
        role_eligibility_schedule_id,
        role_assignment_schedule_id,
    )