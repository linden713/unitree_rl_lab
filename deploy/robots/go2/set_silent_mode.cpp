#include <iostream>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

using namespace unitree::robot;
using namespace unitree::robot::b2;

int main(int argc, const char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " networkInterface [enable|disable]" << std::endl;
        std::cout << "Example: " << argv[0] << " eth0 enable" << std::endl;
        std::cout << "         " << argv[0] << " eth0 disable" << std::endl;
        std::cout << "Default: enable silent mode" << std::endl;
        return -1;
    }
    
    std::string networkInterface = argv[1];
    bool enableSilent = true; 
    
    // check if the network interface is provided
    if (networkInterface == "enable" || networkInterface == "disable") {
        std::cout << "Error: Missing network interface parameter!" << std::endl;
        std::cout << "Usage: " << argv[0] << " networkInterface [enable|disable]" << std::endl;
        std::cout << "Example: " << argv[0] << " eth0 enable" << std::endl;
        return -1;
    }
    
    if (argc > 2) {
        std::string mode = argv[2];
        if (mode == "disable") {
            enableSilent = false;
        } else if (mode != "enable") {
            std::cout << "Warning: Unknown mode '" << mode << "', using default (enable)" << std::endl;
        }
    }
    
    std::cout << "Network Interface: " << networkInterface << std::endl;
    std::cout << "Silent Mode: " << (enableSilent ? "Enable" : "Disable") << std::endl;
    
    try {
        // initialize the channel factory with the specified network interface
        ChannelFactory::Instance()->Init(0, networkInterface);
        
        MotionSwitcherClient msc;
        msc.SetTimeout(10.0f);
        msc.Init();
        
        std::cout << "Connecting to Go2..." << std::endl;
        
        std::string robotForm, motionName;
        int32_t ret = msc.CheckMode(robotForm, motionName);
        
        if (ret == 0) {
            std::cout << "Current status - Robot Form: " << robotForm 
                      << ", Motion Name: " << motionName << std::endl;
            
            if (!motionName.empty()) {
                std::cout << "Motion control service is running. Releasing..." << std::endl;
                ret = msc.ReleaseMode();
                if (ret == 0) {
                    std::cout << "Motion control service released successfully." << std::endl;
                } else {
                    std::cout << "Failed to release motion control service. Error code: " << ret << std::endl;
                }
            } else {
                std::cout << "Motion control service is already stopped." << std::endl;
            }
        } else {
            std::cout << "Failed to check current mode. Error code: " << ret << std::endl;
            return -1;
        }
        
        std::cout << (enableSilent ? "Enabling" : "Disabling") << " silent mode..." << std::endl;
        ret = msc.SetSilent(enableSilent);
        
        if (ret == 0) {
            std::cout << "Silent mode " << (enableSilent ? "enabled" : "disabled") 
                      << " successfully!" << std::endl;
            if (enableSilent) {
                std::cout << "Go2 will not start motion control services automatically after reboot." << std::endl;
            } else {
                std::cout << "Go2 will start motion control services automatically after reboot." << std::endl;
            }
        } else {
            std::cout << "Failed to " << (enableSilent ? "enable" : "disable") 
                      << " silent mode. Error code: " << ret << std::endl;
            
            switch (ret) {
                case 7001:
                    std::cout << "Error: Request parameter error." << std::endl;
                    break;
                case 7002:
                    std::cout << "Error: Switcher service is busy, please try again later." << std::endl;
                    break;
                case 7004:
                    std::cout << "Error: Motion mode name not supported." << std::endl;
                    break;
                default:
                    std::cout << "Error: Unknown error code " << ret << std::endl;
                    break;
            }
            return -1;
        }
        
        bool currentSilent;
        ret = msc.GetSilent(currentSilent);
        if (ret == 0) {
            std::cout << "Verification: Silent mode is currently " 
                      << (currentSilent ? "enabled" : "disabled") << std::endl;
        }
        
        ChannelFactory::Instance()->Release();
        
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}