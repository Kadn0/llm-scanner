// Loaded into promptfoo with NODE_OPTIONS=--require. promptfoo's viewer listens on every network interface; this
// keeps it (and any other server promptfoo starts) on 127.0.0.1, so eval data never reaches the local network.
const net = require("net");
const listen = net.Server.prototype.listen;
net.Server.prototype.listen = function (...args) {
  if (typeof args[0] === "number" || (typeof args[0] === "string" && /^\d+$/.test(args[0]))) {
    if (typeof args[1] !== "string") args.splice(1, 0, "127.0.0.1");
    else args[1] = "127.0.0.1";
  } else if (args[0] && typeof args[0] === "object" && !(args[0] instanceof net.Server) && "port" in args[0]) {
    args[0] = { ...args[0], host: "127.0.0.1" };
  }
  return listen.apply(this, args);
};
