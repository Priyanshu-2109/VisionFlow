"use client";

import { useState } from "react";
import {
  MantineProvider,
  Tooltip,
  UnstyledButton,
  Center,
  Stack,
} from "@mantine/core";
import {
  IconCalendarStats,
  IconDeviceDesktopAnalytics,
  IconFingerprint,
  IconGauge,
  IconHome2,
  IconLogout,
  IconSettings,
  IconSwitchHorizontal,
  IconUser,
  IconSearch,
  IconBell,
  IconMessage,
  IconChartBar,
  IconUsers,
} from "@tabler/icons-react";
import Link from "next/link";

const navLinks = [
  { icon: IconHome2, label: "Home" },
  { icon: IconGauge, label: "Dashboard" },
  { icon: IconDeviceDesktopAnalytics, label: "Analytics" },
  { icon: IconCalendarStats, label: "Releases" },
  { icon: IconUser, label: "Account" },
  { icon: IconFingerprint, label: "Security" },
  { icon: IconSettings, label: "Settings" },
];

function NavbarLink({ icon: Icon, label, active, onClick }) {
  return (
    <UnstyledButton
      onClick={onClick}
      className={`relative w-[50px] h-[50px] flex items-center justify-center rounded-md transition-all
        ${
          active
            ? "bg-blue-500 text-white"
            : "text-gray-400 hover:bg-gray-800 hover:text-white"
        }`}
    >
      {active && (
        <div className="absolute left-0 w-[3px] h-[60%] bg-white rounded-r-md" />
      )}
      <Icon size={22} stroke={1.5} />
    </UnstyledButton>
  );
}

export default function Dashboard() {
  const [active, setActive] = useState(1); // Default to Dashboard

  return (
    <MantineProvider
      withGlobalStyles
      withNormalizeCSS
      theme={{ colorScheme: "dark" }}
    >
      <style jsx global>{`
        ::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }

        ::-webkit-scrollbar-track {
          background: rgb(10, 10, 10);
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
          background: #3b82f6;
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: #2563eb;
        }

        * {
          scrollbar-width: thin;
          scrollbar-color: #3b82f6 rgb(10, 10, 10);
        }
      `}</style>

      <div className="flex h-screen bg-[rgb(10,10,10)] text-white">
        <nav className="w-[80px] h-screen bg-[rgb(15,15,15)] border-r border-gray-800/50 flex flex-col p-4 shadow-xl">
          <Center className="py-4">
            <div className="">
              <Link href={"/"}>
                <img src="/VF_withoutbg.png" alt="Logo" className="w-8 h-8" />
              </Link>
            </div>
          </Center>

          <div className="flex-1 mt-8">
            <Stack justify="center" gap={8}>
              {navLinks.map((link, index) => (
                <NavbarLink
                  key={link.label}
                  {...link}
                  active={index === active}
                  onClick={() => setActive(index)}
                />
              ))}
            </Stack>
          </div>

          <Stack justify="center" gap={8} className="mb-4">
            <div className="w-full h-px bg-gray-800 my-2"></div>
            <NavbarLink icon={IconSwitchHorizontal} label="Change account" />
            <NavbarLink icon={IconLogout} label="Logout" />
          </Stack>
        </nav>

        <div className="flex-1 flex flex-col">
          <header className="h-16 border-b border-gray-800/50 flex items-center justify-between px-8 bg-[rgb(15,15,15)]">
            <div className="text-xl font-semibold">
              {navLinks[active].label}
            </div>

            <div className="flex items-center space-x-6">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search..."
                  className="pl-10 pr-4 py-2 bg-gray-800/50 rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 w-64"
                />
                <IconSearch
                  className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                  size={18}
                />
              </div>

              <div className="flex items-center space-x-4">
                <button className="relative p-2 rounded-lg hover:bg-gray-800 transition-colors">
                  <IconBell size={20} className="text-gray-400" />
                  <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
                </button>
                <button className="relative p-2 rounded-lg hover:bg-gray-800 transition-colors">
                  <IconMessage size={20} className="text-gray-400" />
                  <span className="absolute top-1 right-1 w-2 h-2 bg-blue-500 rounded-full"></span>
                </button>
                <div className="w-px h-8 bg-gray-700 mx-2"></div>
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600"></div>
                  <span className="text-sm font-medium">Alex Parker</span>
                </div>
              </div>
            </div>
          </header>

          <main className="flex-1 p-8 overflow-auto bg-[rgb(10,10,10)]">
            <div className="bg-[rgb(15,15,15)] rounded-xl p-6 mb-8 border border-gray-800/50">
              <h1 className="text-3xl font-bold mb-2">
                Welcome to {navLinks[active].label}
              </h1>
              <p className="text-gray-400 max-w-2xl">
                Manage your business operations, analyze performance metrics,
                and optimize your workflow with our powerful dashboard tools.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {[
                {
                  title: "Total Revenue",
                  value: "₹24,562.00",
                  growth: "+5.2% from last month",
                  icon: IconChartBar,
                },
                {
                  title: "Active Projects",
                  value: "3",
                  growth: "2 new this week",
                  icon: IconCalendarStats,
                },
                {
                  title: "Customer Growth",
                  value: "+12.3%",
                  growth: "248 new customers",
                  icon: IconUsers,
                },
              ].map(({ title, value, growth, icon: Icon }, index) => (
                <div
                  key={index}
                  className="bg-[rgb(15,15,15)] p-6 rounded-xl border border-gray-800/50 shadow-lg"
                >
                  <div className="flex justify-between items-start mb-4">
                    <h3 className="text-lg font-medium">{title}</h3>
                    <div className="p-2 bg-blue-500/20 rounded-lg">
                      <Icon size={20} className="text-blue-400" />
                    </div>
                  </div>
                  <div className="text-3xl font-bold mb-2">{value}</div>
                  <div className="text-blue-400 text-sm">{growth}</div>
                </div>
              ))}
            </div>
            {/* Additional Content Section */}
            <div className="mt-8 bg-[rgb(10,10,10)] rounded-xl p-6 border border-gray-800/50 shadow-lg">
              <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
              <div className="space-y-4">
                {[1, 2, 3].map((item) => (
                  <div
                    key={item}
                    className="flex items-center p-3 hover:bg-gray-900/50 rounded-lg transition-colors cursor-pointer"
                  >
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-gray-700/20 to-gray-600/20 flex items-center justify-center mr-4">
                      <IconDeviceDesktopAnalytics
                        size={20}
                        className="text-gray-400"
                      />
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-gray-300">
                        New analytics report generated
                      </div>
                      <div className="text-sm text-gray-500">
                        Generated comprehensive Q1 performance report
                      </div>
                    </div>
                    <div className="text-sm text-gray-500">2h ago</div>
                  </div>
                ))}
              </div>

              {/* View All Button */}
              <button className="mt-4 w-full py-2 bg-gray-800/50 hover:bg-gray-700/50 text-gray-300 rounded-lg transition-colors">
                View All Activity
              </button>
            </div>
          </main>
        </div>
      </div>
    </MantineProvider>
  );
}
