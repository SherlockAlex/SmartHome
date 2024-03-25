#ifndef SMARTHOME_NET_H
#define SMARTHOME_NET_H

namespace network{
  class Network{
  public:
    void SetupWiFi();
    void ScanWiFi();
    void Connect(const char* ssid,const char* passwd);
  };
}

#endif