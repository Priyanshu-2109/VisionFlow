"use client";
import { usePathname } from "next/navigation";
import { useState } from "react";
import ArrowForwardIosIcon from "@mui/icons-material/ArrowForwardIos";
import {
  IconGauge,
  IconNotes,
  IconCalendarStats,
  IconPresentationAnalytics,
  IconFileAnalytics,
  IconAdjustments,
  IconLock,
} from "@tabler/icons-react";
import Link from "next/link";

const mockdata = [
  { label: "Dashboard", icon: <IconGauge size={20} />, link: "/userdashboard" },
{
    label: "Market news",
    icon: <IconNotes size={20} />,
    links: [
      { label: "Overview", link: "/overview" },
      { label: "Forecasts", link: "/forecasts" },
      { label: "Outlook", link: "/outlook" },
      { label: "Real time", link: "/realtime" },
    ],
  },
  { label: "Analytics", icon: <IconPresentationAnalytics size={20} />, link: "/analytics" },
  { label: "Contracts", icon: <IconFileAnalytics size={20} />, link: "/contracts" },
  { label: "Settings", icon: <IconAdjustments size={20} />, link: "/settings" },
  {
    label: "Security",
    icon: <IconLock size={20} />,
    links: [
      { label: "Enable 2FA", link: "/security/2fa" },
      { label: "Change password", link: "/security/password" },
      { label: "Recovery codes", link: "/security/recovery" },
    ],
  },
];

const SidebarLayout = ({ children }) => {
  const pathname = usePathname();
  const [openMenu, setOpenMenu] = useState({});

  const toggleMenu = (label) => {
    setOpenMenu((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  // Extract the page name from the path
  const pageTitle = mockdata
    .flatMap((item) => [item, ...(item.links || [])])
    .find((item) => item.link === pathname)?.label || "Dashboard";

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <nav className="w-64 bg-gray-900 text-white flex flex-col">
        {/* Sidebar Header */}
        <div className="p-5 flex justify-between items-center">
          <h2 className="text-xl font-bold">VisionFlow</h2>
          <span className="text-sm bg-gray-700 px-2 py-1 rounded">v0.0.1</span>
        </div>

        {/* Sidebar Links */}
        <div className="flex-1 overflow-y-auto">
          {mockdata.map((item) => (
            <div key={item.label} className="p-2">
              <div
                className="flex items-center justify-between p-3 hover:bg-gray-800 cursor-pointer rounded-lg"
                onClick={() => item.links && toggleMenu(item.label)}
              >
                <div className="flex items-center gap-2">
                  {item.icon}
                  {item.link ? (
                    <Link href={item.link} className="text-sm">
                      {item.label}
                    </Link>
                  ) : (
                    <span className="text-sm">{item.label}</span>
                  )}
                </div>
                {item.links && (
                  <ArrowForwardIosIcon
                    className={`transition-transform ${openMenu[item.label] ? "rotate-90" : ""}`}
                    fontSize="x-small"
                  />
                )}
              </div>

              {/* Submenu */}
              <div
                className={`overflow-hidden transition-all duration-500 ease-in-out ${
                  openMenu[item.label] ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0"
                }`}
              >
                {item.links &&
                  item.links.map((sub) => (
                    <Link key={sub.label} href={sub.link} className="block text-sm p-2 hover:bg-gray-800 rounded">
                      {sub.label}
                    </Link>
                  ))}
              </div>
            </div>
          ))}
        </div>

        {/* Sidebar Footer */}
        <div className="p-5 border-t border-gray-700">
          <button className="w-full bg-gray-800 p-3 rounded-lg hover:bg-gray-700">User Settings</button>
        </div>
      </nav>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Top Navbar */}
        <div className="bg-gray-800 text-white p-4 text-xl font-semibold">
          {pageTitle}
        </div>

        {/* Page Content */}
        <div className="flex-1 p-6">{children}</div>
      </div>
    </div>
  );
};

export default SidebarLayout;
