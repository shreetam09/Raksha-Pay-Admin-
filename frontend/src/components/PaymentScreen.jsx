import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft,
  Send,
  Smartphone,
  PhoneCall,
  UserPlus,
  Terminal,
  ShieldCheck,
  Cpu,
  ScanLine
} from 'lucide-react';
import ThemeToggle from './ThemeToggle';

export const PaymentScreen = ({ onTriggerIntervention, onPaymentSuccess, isDark, setIsDark }) => {
  const [amount, setAmount] = useState('25,000');
  const [upiId, setUpiId] = useState('ramesh@upi');
  const [isExtracting, setIsExtracting] = useState(false);
  
  // Real Device Data State
  const [realDeviceData, setRealDeviceData] = useState({
    os: 'Detecting...',
    browser: 'Detecting...',
    battery: 'Detecting...',
    network: 'Detecting...',
    screenResolution: 'Detecting...',
    language: 'Detecting...',
    timezone: 'Detecting...'
  });

  // Synthetic Coercion Toggles (Things we can't extract from a browser easily)
  const [telemetry, setTelemetry] = useState({
    activeCall: false,
    newDevice: false,
    firstTimePayee: false,
  });

  // 1. ACTUAL LOCAL FEATURE EXTRACTION LOGIC
  useEffect(() => {
    const extractDeviceData = async () => {
      // OS Detection
      const ua = navigator.userAgent;
      let os = "Unknown OS";
      if (ua.includes("Win")) os = "Windows";
      if (ua.includes("Mac")) os = "MacOS";
      if (ua.includes("Linux")) os = "Linux";
      if (ua.includes("Android")) os = "Android";
      if (ua.includes("iPhone") || ua.includes("iPad")) os = "iOS";

      // Battery Extraction
      let batteryLevel = "Unsupported";
      if ('getBattery' in navigator) {
        try {
          const battery = await navigator.getBattery();
          batteryLevel = `${Math.round(battery.level * 100)}% ${battery.charging ? '(Charging)' : ''}`;
        } catch (e) {
          batteryLevel = "Access Denied";
        }
      }

      // Network Extraction
      const connection = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
      const networkType = connection ? `${connection.effectiveType.toUpperCase()} (${connection.downlink}Mbps)` : "Unknown";

      setRealDeviceData({
        os,
        browser: navigator.vendor || "Unknown",
        battery: batteryLevel,
        network: networkType,
        screenResolution: `${window.screen.width}x${window.screen.height}`,
        language: navigator.language,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
      });
    };

    extractDeviceData();
  }, []);

  const toggleTelemetry = (key) => {
    setTelemetry((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handlePayment = (e) => {
    e?.preventDefault();
    
    // Simulate the extraction delay before scoring
    setIsExtracting(true);
    
    setTimeout(() => {
      setIsExtracting(false);
      const activeRiskCount = Object.values(telemetry).filter(Boolean).length;

      if (activeRiskCount >= 2) {
        if (onTriggerIntervention) {
          onTriggerIntervention({ amount, upiId, telemetry });
        }
      } else {
        if (onPaymentSuccess) {
          onPaymentSuccess({ amount, upiId });
        }
      }
    }, 1500); // 1.5 second scanning animation
  };

  const liveJSON = {
    timestamp: new Date().toISOString(),
    transaction_features: {
      amount_inr: amount.replace(/,/g, ''),
      payee_id: upiId || 'null',
    },
    extracted_device_hardware: realDeviceData,
    situational_sensors: {
      call_state_active: telemetry.activeCall,
      device_fingerprint_match: !telemetry.newDevice,
      payee_in_contacts: !telemetry.firstTimePayee,
    }
  };

  return (
    <div className="min-h-screen bg-[var(--color-pure-white)] dark:bg-[var(--color-abyss)] flex flex-col lg:flex-row overflow-hidden font-sans transition-colors duration-300">
      
      {/* LEFT COLUMN: The Banking App / Phone Simulator */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-4 lg:p-12 relative bg-[var(--color-off-white-canvas)] dark:bg-[var(--color-obsidian)] transition-colors duration-300">
        
        {/* Neon Accent Glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] bg-[var(--color-electric-lime)] dark:bg-[var(--color-iris-gleam)] rounded-full blur-[150px] opacity-20 pointer-events-none transition-colors duration-300" />

        <motion.div 
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="relative w-full max-w-[400px] min-h-[700px] bg-[var(--color-pure-white)] dark:bg-[var(--color-graphite-dark)] rounded-[32px] border border-[var(--color-ash)] dark:border-[var(--color-steel)] shadow-2xl overflow-hidden flex flex-col transition-colors duration-300"
        >
          {/* Scanning Overlay (Appears when Pay is clicked) */}
          <AnimatePresence>
            {isExtracting && (
              <motion.div 
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="absolute inset-0 z-50 bg-[var(--color-pure-white)]/95 dark:bg-[var(--color-abyss)]/90 backdrop-blur-sm flex flex-col items-center justify-center text-[var(--color-off-black-ink)] dark:text-[var(--color-iris-gleam)]"
              >
                <motion.div 
                  animate={{ y: [-20, 20, -20] }} 
                  transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                >
                  <ScanLine className="w-16 h-16 mb-4 opacity-80" />
                </motion.div>
                <p className="font-mono text-sm tracking-widest font-bold">EXTRACTING LOCAL FEATURES</p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* App Header */}
          <div className="p-6 bg-[var(--color-off-white-canvas)] dark:bg-[#111111] border-b border-[var(--color-ash)] dark:border-[var(--color-steel)] flex items-center justify-between transition-colors duration-300">
            <div className="flex items-center space-x-3">
              <ArrowLeft className="w-5 h-5 text-[var(--color-off-black-ink)] dark:text-[var(--color-cloud)]" />
              <h1 className="text-xl font-bold tracking-tight text-[var(--color-off-black-ink)] dark:text-white">Send Money</h1>
            </div>
            <ShieldCheck className="w-6 h-6 text-[var(--color-electric-lime)] dark:text-[var(--color-iris-gleam)]" />
          </div>

          {/* App Body */}
          <div className="flex-1 p-8 flex flex-col justify-center">
            <form onSubmit={handlePayment} className="space-y-8">
              <div className="space-y-2">
                <label className="block text-xs font-bold text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)] uppercase tracking-wider">
                  Payee UPI ID
                </label>
                <input
                  type="text"
                  value={upiId}
                  onChange={(e) => setUpiId(e.target.value)}
                  className="w-full py-4 px-4 rounded-xl border border-[var(--color-ash)] dark:border-[var(--color-steel)] text-lg font-semibold text-[var(--color-off-black-ink)] dark:text-white bg-[var(--color-off-white-canvas)] dark:bg-[var(--color-abyss)] focus:outline-none focus:border-[var(--color-electric-lime)] dark:focus:border-[var(--color-iris-gleam)] transition-colors"
                />
              </div>

              <div className="space-y-2">
                <label className="block text-xs font-bold text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)] uppercase tracking-wider text-center">
                  Amount
                </label>
                <div className="flex items-center justify-center py-6">
                  <span className="text-4xl font-extrabold text-[var(--color-off-black-ink)] dark:text-white mr-2">₹</span>
                  <input
                    type="text"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    className="w-48 text-5xl font-extrabold text-[var(--color-off-black-ink)] dark:text-white bg-transparent focus:outline-none text-left"
                  />
                </div>
              </div>

              <motion.button
                whileTap={{ scale: 0.95 }}
                type="submit"
                className="w-full py-4 rounded-[var(--radius-full)] bg-[var(--color-electric-lime)] dark:bg-[var(--color-iris-gleam)] text-[var(--color-off-black-ink)] dark:text-white font-bold text-lg shadow-[0_0_20px_rgba(190,255,80,0.3)] dark:shadow-[0_0_20px_rgba(132,125,255,0.3)] flex items-center justify-center space-x-2 transition-all hover:shadow-[0_0_30px_rgba(190,255,80,0.5)] dark:hover:shadow-[0_0_30px_rgba(132,125,255,0.5)] cursor-pointer"
              >
                <span>Pay Securely</span>
                <Send className="w-5 h-5" />
              </motion.button>
            </form>
          </div>
        </motion.div>
      </div>

      {/* RIGHT COLUMN: Real-Time Feature Terminal */}
      <div className="w-full lg:w-1/2 bg-[var(--color-pure-white)] dark:bg-[#090a0b] border-l border-[var(--color-ash)] dark:border-gray-800 p-6 lg:p-12 flex flex-col h-full transition-colors duration-300">
        
        <div className="flex items-center justify-between gap-3 mb-8">
          <div className="flex items-center gap-3">
            <Cpu className="w-8 h-8 text-[var(--color-electric-lime)] dark:text-[var(--color-iris-gleam)]" />
            <div>
              <h2 className="text-2xl font-bold text-[var(--color-off-black-ink)] dark:text-white tracking-tight">On-Device Extraction</h2>
              <p className="text-sm text-[var(--color-graphite)] dark:text-gray-400">Real-time local hardware and environmental state.</p>
            </div>
          </div>
          <ThemeToggle isDark={isDark} setIsDark={setIsDark} />
        </div>

        {/* Presenter Overrides */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {[
            { id: 'activeCall', label: 'Call Active', icon: PhoneCall },
            { id: 'newDevice', label: 'New Device', icon: Smartphone },
            { id: 'firstTimePayee', label: 'New Payee', icon: UserPlus }
          ].map((item) => (
            <div 
              key={item.id}
              onClick={() => toggleTelemetry(item.id)}
              className={`p-3 rounded-lg border cursor-pointer transition-all flex items-center justify-center gap-2 text-sm font-bold ${
                telemetry[item.id] 
                ? 'bg-[var(--color-electric-lime)] dark:bg-[var(--color-iris-gleam)] border-[var(--color-electric-lime)] dark:border-[var(--color-iris-gleam)] text-[var(--color-off-black-ink)] dark:text-white' 
                : 'bg-[var(--color-off-white-canvas)] dark:bg-[var(--color-graphite-dark)] border-[var(--color-ash)] dark:border-[var(--color-steel)] text-[var(--color-graphite)] dark:text-[var(--color-ash-dark)] hover:border-[var(--color-off-black-ink)] dark:hover:border-gray-600'
              }`}
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </div>
          ))}
        </div>

        {/* Syntax Highlighted JSON Terminal */}
        <div className="flex-1 bg-[var(--color-off-white-canvas)] dark:bg-[#141414] rounded-xl border border-[var(--color-ash)] dark:border-gray-800 overflow-hidden flex flex-col font-mono shadow-2xl transition-colors duration-300">
          <div className="bg-[var(--color-ash)]/30 dark:bg-[#1a1a1a] px-4 py-3 border-b border-[var(--color-ash)] dark:border-gray-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-[var(--color-graphite)] dark:text-gray-400" />
              <span className="text-[var(--color-graphite)] dark:text-gray-400 text-xs tracking-wider">feature_vector.json</span>
            </div>
            <div className="flex gap-2">
              <span className="w-3 h-3 rounded-full bg-red-500/80"></span>
              <span className="w-3 h-3 rounded-full bg-yellow-500/80"></span>
              <span className="w-3 h-3 rounded-full bg-green-500/80"></span>
            </div>
          </div>
          
          <div className="p-6 overflow-y-auto text-sm leading-relaxed flex-1">
            <pre className="text-[var(--color-off-black-ink)] dark:text-gray-300">
              <span className="text-[#847dff]">{'{'}</span>
              <br/>
              <span className="text-[#00b3dd]">  "timestamp"</span>: <span className="text-[#beff50]">"{liveJSON.timestamp}"</span>,
              <br/>
              <span className="text-[#00b3dd]">  "transaction_features"</span>: <span className="text-[#847dff]">{'{'}</span>
              <br/>
              <span className="text-[#00b3dd]">    "amount_inr"</span>: <span className="text-[#dd90d8]">{liveJSON.transaction_features.amount_inr}</span>,
              <br/>
              <span className="text-[#00b3dd]">    "payee_id"</span>: <span className="text-[#beff50]">"{liveJSON.transaction_features.payee_id}"</span>
              <br/>
              <span className="text-[#847dff]">  {'}'}</span>,
              <br/>
              <span className="text-[#00b3dd]">  "extracted_device_hardware"</span>: <span className="text-[#847dff]">{'{'}</span>
              <br/>
              <span className="text-[#00b3dd]">    "os"</span>: <span className="text-[#beff50]">"{liveJSON.extracted_device_hardware.os}"</span>,
              <br/>
              <span className="text-[#00b3dd]">    "battery"</span>: <span className="text-[#beff50]">"{liveJSON.extracted_device_hardware.battery}"</span>,
              <br/>
              <span className="text-[#00b3dd]">    "network"</span>: <span className="text-[#beff50]">"{liveJSON.extracted_device_hardware.network}"</span>,
              <br/>
              <span className="text-[#00b3dd]">    "resolution"</span>: <span className="text-[#beff50]">"{liveJSON.extracted_device_hardware.screenResolution}"</span>
              <br/>
              <span className="text-[#847dff]">  {'}'}</span>,
              <br/>
              <span className="text-[#00b3dd]">  "situational_sensors"</span>: <span className="text-[#847dff]">{'{'}</span>
              <br/>
              <span className="text-[#00b3dd]">    "call_state_active"</span>: <span className={liveJSON.situational_sensors.call_state_active ? "text-[#ff4433]" : "text-[#dd90d8]"}>{String(liveJSON.situational_sensors.call_state_active)}</span>,
              <br/>
              <span className="text-[#00b3dd]">    "device_match"</span>: <span className={liveJSON.situational_sensors.device_fingerprint_match ? "text-[#dd90d8]" : "text-[#ff4433]"}>{String(liveJSON.situational_sensors.device_fingerprint_match)}</span>,
              <br/>
              <span className="text-[#00b3dd]">    "payee_in_contacts"</span>: <span className={liveJSON.situational_sensors.payee_in_contacts ? "text-[#dd90d8]" : "text-[#ff4433]"}>{String(liveJSON.situational_sensors.payee_in_contacts)}</span>
              <br/>
              <span className="text-[#847dff]">  {'}'}</span>
              <br/>
              <span className="text-[#847dff]">{'}'}</span>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PaymentScreen;