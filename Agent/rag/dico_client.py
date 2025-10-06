"""
DICO API Client for fetching database schema from Efficy CRM.
"""
import logging
import requests

from ..config import get_settings

settings = get_settings()


class DicoAPI:
    """Client for fetching database schema from Efficy DICO API."""
    
    def __init__(self, base_url=None, customer=None):
        self.base_url = base_url or settings.efficy_base_url
        self.customer = customer or settings.efficy_customer

    def fetch_database_schema(self):
        """
        Fetch the complete database schema from DICO API.
        
        Returns:
            dict: Database schema JSON or None if failed
        """
        session = requests.Session() 

        try:
            login_response = session.post( 
                f"{self.base_url}/crm/logon",
                headers={
                    'X-Efficy-Customer': self.customer,
                    'X-Requested-By': 'User',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                data=f'user={settings.efficy_username}&password={settings.efficy_password}'
            )

            if login_response.status_code == 200:
                logging.info("DICO's login successful.")

                dico_response = session.get(
                    f"{self.base_url}/crm/system/dico",
                    headers={
                        'X-Requested-By': 'User',
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                )

                if dico_response.status_code == 200:
                    logging.info("DICO data retrieved successfully.")
                    return dico_response.json()
                else: 
                    logging.error(f"Failed to retrieve DICO data: {dico_response.status_code} - {dico_response.text}")

            else: 
                logging.error(f"Login failed: {login_response.status_code} - {login_response.text}")

        except Exception as e:
            logging.error(f"An error occurred during DICO API interaction: {str(e)}")
        
        return None
