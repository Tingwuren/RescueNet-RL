/*
 * disaster-pro.cc - 应急通信联合救援数字孪生平台
 * 仿真时长: 100秒, 灾害触发: 50秒
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-module.h"

#include <fstream>
#include <vector>
#include <map>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DisasterPro");

// ========================================================================
// 全局配置
// ========================================================================

struct Config {
    std::string deploymentFile = "deployment.json";
    double mapWidth = 5000.0;
    double mapHeight = 5000.0;
    double simTime = 10.0;
    double disasterTime = 5.0;
    double checkInterval = 1.0;
    double wifiRange = 150.0;
    double lteRange = 500.0;
    uint32_t minTaskSize = 256 * 1024;
    uint32_t maxTaskSize = 2 * 1024 * 1024;
    uint32_t maxConcurrentTasks = 50;
    double wifiSpeed = 10.0 * 1024 * 1024;
    double lteSpeed = 5.0 * 1024 * 1024;
} g_config;

struct NodeInfo {
    uint32_t id;
    int type;
    double x, y;
    bool isOnline;
    uint64_t rxBytes;
};

struct ActiveTask {
    uint32_t srcId, dstId;
    double startTime, endTime;
    std::string protocol;
    uint32_t taskSize;
    uint64_t bytesSent;
};

std::vector<NodeInfo> g_nodes;
std::map<std::string, ActiveTask> g_activeTasks;
uint64_t g_totalTxPackets = 0;
uint64_t g_totalRxPackets = 0;
uint64_t g_totalRxBytes = 0;
bool g_disasterOccurred = false;
std::ofstream g_traceFile;

uint32_t g_userCount = 0, g_macroCount = 0, g_manpackCount = 0;
uint32_t g_smallCellCount = 0, g_relayCount = 0;

NetDeviceContainer g_ueDevices;
NetDeviceContainer g_macroEnbDevices;
NetDeviceContainer g_manpackEnbDevices;
NetDeviceContainer g_smallCellDevices;
Ptr<LteHelper> g_lteHelper;
Ipv4InterfaceContainer g_ueIpIfaces;
std::vector<Ipv4Address> g_ueIpv4Addresses;

// ========================================================================
// JSON 解析
// ========================================================================

std::string Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

std::string ExtractValue(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";

    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    pos++;

    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        size_t endPos = json.find('"', pos + 1);
        return json.substr(pos + 1, endPos - pos - 1);
    } else if (json[pos] == '[' || json[pos] == '{') {
        char bracket = json[pos];
        size_t count = 1;
        size_t endPos = pos + 1;
        while (endPos < json.size() && count > 0) {
            if (json[endPos] == bracket) count++;
            else if (json[endPos] == (bracket == '[' ? ']' : '}')) count--;
            endPos++;
        }
        return json.substr(pos, endPos - pos);
    } else {
        size_t endPos = pos;
        while (endPos < json.size() && json[endPos] != ',' && json[endPos] != '}' && json[endPos] != ']') endPos++;
        return Trim(json.substr(pos, endPos - pos));
    }
}

bool ParseDeploymentJson(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << filename << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();

    std::string widthStr = ExtractValue(content, "map_width");
    if (!widthStr.empty()) { try { g_config.mapWidth = std::stod(widthStr); } catch (...) {} }

    std::string heightStr = ExtractValue(content, "map_height");
    if (!heightStr.empty()) { try { g_config.mapHeight = std::stod(heightStr); } catch (...) {} }

    std::string nodesStr = ExtractValue(content, "nodes");
    if (nodesStr.empty()) { std::cerr << "Missing nodes array" << std::endl; return false; }

    if (nodesStr.front() == '[' && nodesStr.back() == ']') {
        nodesStr = nodesStr.substr(1, nodesStr.size() - 2);
    }

    int braceCount = 0;
    std::string currentObj;

    for (size_t i = 0; i < nodesStr.size(); i++) {
        char c = nodesStr[i];

        if (c == '{') {
            braceCount++;
            currentObj += c;
        } else if (c == '}') {
            braceCount--;
            currentObj += c;

            if (braceCount == 0 && !currentObj.empty()) {
                NodeInfo node;

                std::string idStr = ExtractValue(currentObj, "id");
                if (!idStr.empty()) { try { node.id = std::stoi(idStr); } catch (...) { node.id = 0; } }

                std::string xStr = ExtractValue(currentObj, "x");
                if (!xStr.empty()) { try { node.x = std::stod(xStr); } catch (...) { node.x = 0; } }

                std::string yStr = ExtractValue(currentObj, "y");
                if (!yStr.empty()) { try { node.y = std::stod(yStr); } catch (...) { node.y = 0; } }

                std::string typeStr = ExtractValue(currentObj, "type");
                if (typeStr == "USER") { node.type = 0; node.isOnline = true; g_userCount++; }
                else if (typeStr == "MACRO_ENB") { node.type = 1; node.isOnline = true; g_macroCount++; }
                else if (typeStr == "MANPACK_ENB") { node.type = 2; node.isOnline = false; g_manpackCount++; }
                else if (typeStr == "SMALL_CELL") { node.type = 3; node.isOnline = true; g_smallCellCount++; }
                else if (typeStr == "RELAY_NODE") { node.type = 4; node.isOnline = false; g_relayCount++; }
                else { node.type = 0; node.isOnline = true; g_userCount++; }

                node.rxBytes = 0;
                g_nodes.push_back(node);
                currentObj.clear();
            }
        } else if (braceCount > 0) {
            currentObj += c;
        }
    }

    return true;
}

// ========================================================================
// 空间索引
// ========================================================================

class SpatialIndex {
    double m_cellSize;
    uint32_t m_gridW, m_gridH;
    std::map<uint64_t, std::vector<uint32_t>> m_cells;
public:
    SpatialIndex(double mapW, double mapH, double cellSize) : m_cellSize(cellSize) {
        m_gridW = static_cast<uint32_t>(std::ceil(mapW / cellSize));
        m_gridH = static_cast<uint32_t>(std::ceil(mapH / cellSize));
        if (m_gridW == 0) m_gridW = 1;
        if (m_gridH == 0) m_gridH = 1;
    }
    void Clear() { m_cells.clear(); }
    void Insert(uint32_t id, double x, double y) {
        uint32_t cx = std::min(static_cast<uint32_t>(x / m_cellSize), m_gridW - 1);
        uint32_t cy = std::min(static_cast<uint32_t>(y / m_cellSize), m_gridH - 1);
        m_cells[cy * m_gridW + cx].push_back(id);
    }
    std::vector<uint32_t> Query(double x, double y, double range) {
        std::vector<uint32_t> result;
        int nCells = static_cast<int>(std::ceil(range / m_cellSize));
        uint32_t cx = static_cast<uint32_t>(x / m_cellSize);
        uint32_t cy = static_cast<uint32_t>(y / m_cellSize);

        for (int dx = -nCells; dx <= nCells; ++dx) {
            for (int dy = -nCells; dy <= nCells; ++dy) {
                int nx = static_cast<int>(cx) + dx;
                int ny = static_cast<int>(cy) + dy;
                if (nx < 0 || nx >= static_cast<int>(m_gridW)) continue;
                if (ny < 0 || ny >= static_cast<int>(m_gridH)) continue;
                auto it = m_cells.find(static_cast<uint32_t>(ny) * m_gridW + static_cast<uint32_t>(nx));
                if (it != m_cells.end()) {
                    result.insert(result.end(), it->second.begin(), it->second.end());
                }
            }
        }
        return result;
    }
};

SpatialIndex* g_spatialIndex = nullptr;

std::string GetLinkKey(uint32_t s, uint32_t d) { return std::to_string(s) + "-" + std::to_string(d); }

double PredictConnectionTime(Ptr<Node> n1, Ptr<Node> n2, double range) {
    auto mob1 = n1->GetObject<MobilityModel>();
    auto mob2 = n2->GetObject<MobilityModel>();
    if (!mob1 || !mob2) return 0.0;

    Vector p1 = mob1->GetPosition(), v1 = mob1->GetVelocity();
    Vector p2 = mob2->GetPosition(), v2 = mob2->GetVelocity();

    Vector dp = Vector(p1.x - p2.x, p1.y - p2.y, 0);
    Vector dv = Vector(v1.x - v2.x, v1.y - v2.y, 0);

    double a = dv.x * dv.x + dv.y * dv.y;
    double b = 2 * (dp.x * dv.x + dp.y * dv.y);
    double c = dp.x * dp.x + dp.y * dp.y - range * range;

    if (std::abs(a) < 1e-6) return (c <= 0) ? 10.0 : 0.0;

    double delta = b * b - 4 * a * c;
    if (delta < 0) return 10.0;

    double t1 = (-b - std::sqrt(delta)) / (2 * a);
    double t2 = (-b + std::sqrt(delta)) / (2 * a);

    if (t2 < 0) return 0.0;
    if (t1 < 0) return t2;
    return std::min(t1, 10.0);
}

// ========================================================================
// 灾害触发
// ========================================================================

void TriggerDisaster() {
    if (g_disasterOccurred) return;
    g_disasterOccurred = true;

    std::cout << "========================================" << std::endl;
    std::cout << "!!! DISASTER AT " << Simulator::Now().GetSeconds() << "s !!!" << std::endl;
    std::cout << "========================================" << std::endl;

    uint32_t offlineMacro = 0, activeManpack = 0;

    for (auto& node : g_nodes) {
        if (node.type == 1) { node.isOnline = false; offlineMacro++; }
        if (node.type == 2) { node.isOnline = true; activeManpack++; }
    }

    std::cout << "  -> Macro offline: " << offlineMacro << std::endl;
    std::cout << "  -> Manpack online: " << activeManpack << std::endl;
    std::cout << "  -> Disaster transition complete (no LTE handover)" << std::endl;
}

// ========================================================================
// 网络控制器
// ========================================================================

void NetworkController(NodeContainer nodes,
                       std::vector<Ptr<Socket>>& wifiSockets,
                       std::vector<Ptr<Socket>>& lteSockets) {

    double now = Simulator::Now().GetSeconds();

    for (auto it = g_activeTasks.begin(); it != g_activeTasks.end();) {
        const ActiveTask& t = it->second;
        if (!g_nodes[t.srcId].isOnline || !g_nodes[t.dstId].isOnline || now >= t.endTime) {
            g_activeTasks.erase(it++);
            continue;
        }
        ++it;
    }

    if (g_activeTasks.size() < g_config.maxConcurrentTasks) {
        g_spatialIndex->Clear();
        for (uint32_t i = 0; i < g_nodes.size(); ++i) {
            if (g_nodes[i].isOnline) {
                Ptr<MobilityModel> mob = nodes.Get(i)->GetObject<MobilityModel>();
                if (mob) g_spatialIndex->Insert(i, mob->GetPosition().x, mob->GetPosition().y);
            }
        }

        std::vector<uint32_t> candidates;
        for (uint32_t i = 0; i < g_nodes.size(); ++i) {
            if (g_nodes[i].type == 0 && g_nodes[i].isOnline) candidates.push_back(i);
        }

        if (!candidates.empty()) {
            std::shuffle(candidates.begin(), candidates.end(),
                        std::mt19937(std::chrono::system_clock::now().time_since_epoch().count()));

            uint32_t attempts = std::min(uint32_t(5), (uint32_t)candidates.size());

            for (uint32_t a = 0; a < attempts && g_activeTasks.size() < g_config.maxConcurrentTasks; ++a) {
                uint32_t srcId = candidates[a];
                Ptr<MobilityModel> srcMob = nodes.Get(srcId)->GetObject<MobilityModel>();
                if (!srcMob) continue;

                auto cands = g_spatialIndex->Query(srcMob->GetPosition().x,
                                                   srcMob->GetPosition().y,
                                                   g_config.lteRange);
                std::shuffle(cands.begin(), cands.end(),
                            std::mt19937(std::chrono::system_clock::now().time_since_epoch().count()));

                for (uint32_t dstId : cands) {
                    if (dstId == srcId || g_nodes[dstId].type != 0 || !g_nodes[dstId].isOnline) continue;
                    if (g_activeTasks.count(GetLinkKey(srcId, dstId))) continue;

                    Ptr<MobilityModel> dstMob = nodes.Get(dstId)->GetObject<MobilityModel>();
                    if (!dstMob) continue;

                    double dist = srcMob->GetDistanceFrom(dstMob);
                    std::string proto;

                    if (dist < g_config.wifiRange && rand() % 3 != 0) {
                        proto = "WIFI";
                    } else if (dist < g_config.lteRange) {
                        proto = "LTE";
                    } else continue;

                    double connTime = PredictConnectionTime(nodes.Get(srcId), nodes.Get(dstId), g_config.wifiRange);
                    if (connTime < 0.5) continue;

                    ActiveTask task;
                    task.srcId = srcId;
                    task.dstId = dstId;
                    task.startTime = now;
                    task.endTime = now + connTime;
                    task.protocol = proto;
                    task.taskSize = g_config.minTaskSize + rand() % (g_config.maxTaskSize - g_config.minTaskSize);
                    task.bytesSent = 0;

                    g_activeTasks[GetLinkKey(srcId, dstId)] = task;
                    break;
                }
            }
        }
    }

    Simulator::Schedule(Seconds(g_config.checkInterval), &NetworkController, nodes, wifiSockets, lteSockets);
}

// ========================================================================
// 数据包统计
// ========================================================================

void ReceivePacketCallback(Ptr<Socket> socket) {
    Ptr<Packet> packet;
    while ((packet = socket->Recv())) {
        g_totalRxPackets++;
        g_totalRxBytes += packet->GetSize();
        uint32_t id = socket->GetNode()->GetId();
        if (id < g_nodes.size()) g_nodes[id].rxBytes += packet->GetSize();
    }
}

// ========================================================================
// 轨迹输出
// ========================================================================

void TraceState(NodeContainer nodes) {
    static double lastTime = 0;
    static uint64_t lastRx = 0;
    static int frameCount = 0;

    double now = Simulator::Now().GetSeconds();
    double dt = now - lastTime;
    double throughput = 0;

    g_totalTxPackets = g_activeTasks.size();

    if (dt > 0.1 && lastTime > 0) {
        throughput = (g_totalRxBytes - lastRx) * 8.0 / (dt * 1000000.0);
        lastTime = now;
        lastRx = g_totalRxBytes;
    }

    if (g_traceFile.is_open()) {
        if (frameCount > 0) g_traceFile << ",";

        g_traceFile << "{\"time\":" << std::fixed << std::setprecision(1) << now
                    << ",\"tp\":" << std::fixed << std::setprecision(2) << throughput
                    << ",\"loss\":" << std::fixed << std::setprecision(2) << 0.0
                    << ",\"disaster\":" << (g_disasterOccurred ? 1 : 0)
                    << ",\"nodes\":[";

        bool first = true;
        for (uint32_t i = 0; i < nodes.GetN(); ++i) {
            Ptr<MobilityModel> mob = nodes.Get(i)->GetObject<MobilityModel>();
            if (!mob) continue;
            Vector pos = mob->GetPosition();
            if (!first) g_traceFile << ",";
            g_traceFile << "[" << i << "," << g_nodes[i].type << ","
                        << (int)pos.x << "," << (int)pos.y << ","
                        << (g_nodes[i].isOnline ? 1 : 0) << "," << g_nodes[i].rxBytes << "]";
            first = false;
        }

        g_traceFile << "],\"links\":[";
        first = true;
        for (const auto& kv : g_activeTasks) {
            const ActiveTask& t = kv.second;
            if (!first) g_traceFile << ",";
            g_traceFile << "[" << t.srcId << "," << t.dstId << "," << (t.protocol == "WIFI" ? 0 : 1) << "]";
            first = false;
        }
        g_traceFile << "]}";
        g_traceFile.flush();

        frameCount++;
    }

    Simulator::Schedule(Seconds(g_config.checkInterval), &TraceState, nodes);
}

// ========================================================================
// 主函数
// ========================================================================

int main(int argc, char* argv[]) {
    CommandLine cmd;
    cmd.AddValue("simTime", "Simulation time", g_config.simTime);
    cmd.AddValue("disasterTime", "Disaster time", g_config.disasterTime);
    cmd.AddValue("config", "Config file", g_config.deploymentFile);
    cmd.Parse(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "   应急通信数字孪生平台 - NS-3" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[1] 加载部署配置..." << std::endl;
    if (!ParseDeploymentJson(g_config.deploymentFile)) return 1;

    std::cout << "    地图: " << g_config.mapWidth << "x" << g_config.mapHeight << std::endl;
    std::cout << "    节点总数: " << g_nodes.size() << std::endl;
    std::cout << "    用户:" << g_userCount << " 宏站:" << g_macroCount
              << " 背负:" << g_manpackCount << " 小站:" << g_smallCellCount << " 中继:" << g_relayCount << std::endl;

    if (g_nodes.empty()) {
        std::cerr << "No nodes loaded! Check deployment.json" << std::endl;
        return 1;
    }

    g_spatialIndex = new SpatialIndex(g_config.mapWidth, g_config.mapHeight, g_config.wifiRange);

    std::cout << "\n[2] 创建节点..." << std::endl;
    NodeContainer allNodes; allNodes.Create(g_nodes.size());
    NodeContainer macroNodes, manpackNodes, smallCellNodes, relayNodes, ueNodes;

    for (const auto& n : g_nodes) {
        if (n.type == 1) macroNodes.Add(allNodes.Get(n.id));
        else if (n.type == 2) manpackNodes.Add(allNodes.Get(n.id));
        else if (n.type == 3) smallCellNodes.Add(allNodes.Get(n.id));
        else if (n.type == 4) relayNodes.Add(allNodes.Get(n.id));
        else ueNodes.Add(allNodes.Get(n.id));
    }

    std::cout << "\n[3] 配置移动模型..." << std::endl;
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    Ptr<ListPositionAllocator> enbPos = CreateObject<ListPositionAllocator>();
    for (const auto& n : g_nodes) {
        if (n.type >= 1 && n.type <= 4) {
            double z = (n.type == 1) ? 30.0 : 15.0;
            enbPos->Add(Vector(n.x, n.y, z));
        }
    }
    mobility.SetPositionAllocator(enbPos);
    mobility.Install(macroNodes);
    mobility.Install(manpackNodes);
    mobility.Install(smallCellNodes);
    mobility.Install(relayNodes);

    MobilityHelper userMob;
    userMob.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                              "Bounds", RectangleValue(Rectangle(0, g_config.mapWidth, 0, g_config.mapHeight)),
                              "Speed", StringValue("ns3::UniformRandomVariable[Min=1.0|Max=5.0]"),
                              "Mode", StringValue("Time"), "Time", StringValue("2s"));
    Ptr<ListPositionAllocator> userPos = CreateObject<ListPositionAllocator>();
    for (const auto& n : g_nodes) if (n.type == 0) userPos->Add(Vector(n.x, n.y, 0));
    userMob.SetPositionAllocator(userPos);
    userMob.Install(ueNodes);

    // ============ 关键修改：先安装 Internet 栈，再配置 LTE ============

    std::cout << "\n[4] 安装 Internet 协议栈..." << std::endl;
    InternetStackHelper internet;
    internet.Install(allNodes);

    // ============ 配置 WiFi (在 LTE 之前) ============
    std::cout << "\n[5] 配置WiFi..." << std::endl;
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::IdealWifiManager");
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(YansWifiChannelHelper::Default().Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(20.0));
    wifiPhy.Set("TxPowerEnd", DoubleValue(20.0));
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer wifiDevs = wifi.Install(wifiPhy, wifiMac, allNodes);

    // WiFi IP 地址
    Ipv4AddressHelper ipWifi;
    ipWifi.SetBase("192.168.0.0", "255.255.0.0");
    Ipv4InterfaceContainer wifiIfaces = ipWifi.Assign(wifiDevs);

    // ============ 配置 LTE (在 Internet 栈已安装之后) ============
    std::cout << "\n[6] 配置LTE..." << std::endl;
    g_lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    g_lteHelper->SetEpcHelper(epcHelper);
    g_lteHelper->SetAttribute("UseIdealRrc", BooleanValue(true));

    if (macroNodes.GetN() > 0) {
        g_macroEnbDevices = g_lteHelper->InstallEnbDevice(macroNodes);
    }
    if (manpackNodes.GetN() > 0) {
        g_manpackEnbDevices = g_lteHelper->InstallEnbDevice(manpackNodes);
    }
    if (smallCellNodes.GetN() > 0) {
        g_smallCellDevices = g_lteHelper->InstallEnbDevice(smallCellNodes);
    }

    // 安装 UE 设备（在 Internet 栈已安装之后）
    g_ueDevices = g_lteHelper->InstallUeDevice(ueNodes);

    // 分配 UE IP 地址
    g_ueIpIfaces = epcHelper->AssignUeIpv4Address(g_ueDevices);
    g_ueIpv4Addresses.resize(g_nodes.size());
    for (uint32_t i = 0; i < g_nodes.size(); ++i) {
        if (g_nodes[i].type == 0) g_ueIpv4Addresses[i] = g_ueIpIfaces.GetAddress(i);
    }

    // Attach 到基站
    if (g_macroEnbDevices.GetN() > 0) {
        for (uint32_t i = 0; i < g_ueDevices.GetN(); ++i) {
            g_lteHelper->Attach(g_ueDevices.Get(i), g_macroEnbDevices.Get(i % g_macroEnbDevices.GetN()));
        }
    }

    // ============ 创建 Socket ============
    std::cout << "\n[7] 创建Socket..." << std::endl;
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    Config::SetDefault("ns3::UdpSocket::RcvBufSize", UintegerValue(16 * 1024 * 1024));

    std::vector<Ptr<Socket>> wifiSockets, lteSockets;
    for (uint32_t i = 0; i < allNodes.GetN(); ++i) {
        Ptr<Socket> ws = Socket::CreateSocket(allNodes.Get(i), tid);
        ws->Bind(InetSocketAddress(wifiIfaces.GetAddress(i), 8080));
        ws->SetRecvCallback(MakeCallback(&ReceivePacketCallback));
        wifiSockets.push_back(ws);

        Ptr<Socket> ls = Socket::CreateSocket(allNodes.Get(i), tid);
        ls->Bind(InetSocketAddress(g_ueIpv4Addresses[i], 8080));
        ls->SetRecvCallback(MakeCallback(&ReceivePacketCallback));
        lteSockets.push_back(ls);
    }

    // ============ 打开追踪文件 ============
    std::cout << "\n[8] 打开追踪文件..." << std::endl;
    g_traceFile.open("trace.json");
    if (!g_traceFile.is_open()) {
        std::cerr << "Failed to open trace.json!" << std::endl;
        return 1;
    }
    g_traceFile << "[";
    g_traceFile.flush();

    // ============ 调度 ============
    std::cout << "\n[9] 调度事件..." << std::endl;
    std::cout << "    仿真时长: " << g_config.simTime << "s" << std::endl;
    std::cout << "    灾害时间: " << g_config.disasterTime << "s" << std::endl;

    Simulator::Schedule(Seconds(g_config.disasterTime), &TriggerDisaster);
    Simulator::Schedule(Seconds(1.0), &TraceState, allNodes);
    Simulator::Schedule(Seconds(2.0), &NetworkController, allNodes, wifiSockets, lteSockets);

    std::cout << "\n[10] 运行仿真..." << std::endl;
    Simulator::Stop(Seconds(g_config.simTime));
    Simulator::Run();

    if (g_traceFile.is_open()) {
        g_traceFile << "]";
        g_traceFile.flush();
        g_traceFile.close();
    }

    Simulator::Destroy();
    delete g_spatialIndex;

    std::cout << "\n========================================" << std::endl;
    std::cout << "   仿真完成!" << std::endl;
    std::cout << "   活跃链路峰值: " << g_activeTasks.size() << std::endl;
    std::cout << "   接收包: " << g_totalRxPackets << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}