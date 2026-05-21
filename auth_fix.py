from google_auth_oauthlib.flow import InstalledAppFlow

flow = InstalledAppFlow.from_client_secrets_file(
    'credentials.json',
    ['https://www.googleapis.com/auth/gmail.send',
     'https://www.googleapis.com/auth/gmail.readonly']
)
creds = flow.run_local_server(port=0)
open('token_send.json', 'w').write(creds.to_json())
print('Done! token_send.json saved.')