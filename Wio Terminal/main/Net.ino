#include "Net.h"
#include "rpcWiFi.h"

void network::Network::SetupWiFi(){
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);
  Serial.println("wifi setup done");
}

void network::Network::Connect(const char* ssid,const char* passwd){
  WiFi.begin(ssid,passwd);
  while(WiFi.status()!=WL_CONNECTED)
  {
    Serial.println("Connecting WiFi");
    WiFi.begin(ssid,passwd);
    delay(500);
  }
  Serial.print("Connected WiFi");
}