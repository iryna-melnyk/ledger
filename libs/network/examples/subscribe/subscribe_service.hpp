#pragma once

#include "./protocols/fetch_protocols.hpp"  // defines enum
#include "./protocols/subscribe/protocol.hpp"
#include "core/logger.hpp"
#include "network/service/server.hpp"
#include <memory>

namespace fetch {
namespace subscribe {


/*
 * Class containing one or more protocols, thus defining a service. Inherits
 * from ServiceServer to provide network functionality
 */
class SubscribeService : public service::ServiceServer<fetch::network::TCPServer>
{
public:
  /*
   * Constructor for SubscribeService, will create a server to respond to rpc
   * calls
   */
  SubscribeService(fetch::network::NetworkManager tm, uint16_t tcpPort) : ServiceServer(tcpPort, tm)
  {
    // Macro used for debugging
    LOG_STACK_TRACE_POINT;

    // Prints when compiled in debug mode. Options: logger.Debug logger.Info
    // logger.Error
    fetch::logger.Debug("Constructing test node service with TCP port: ", tcpPort);

    // We construct our node, and attach it to the protocol
    subscribeProto_ = std::make_unique<protocols::SubscribeProtocol>();

    // We 'Add' these protocols
    this->Add(protocols::FetchProtocols::SUBSCRIBE_PROTO, subscribeProto_.get());
  }

  // We can use this to send messages to interested nodes
  void SendMessage(std::string const &mes) { subscribeProto_->SendMessage(mes); }

private:
  std::unique_ptr<protocols::SubscribeProtocol> subscribeProto_;
};
}  // namespace subscribe
}  // namespace fetch
