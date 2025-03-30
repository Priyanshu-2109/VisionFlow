"use client";

import { useState } from 'react';
import Link from 'next/link';
import {
  IconCalendarStats,
  IconDeviceDesktopAnalytics,
  IconFingerprint,
  IconGauge,
  IconHome2,
  IconSettings,
  IconUser,
} from '@tabler/icons-react';

const mainLinksMockdata = [
  { icon: IconHome2, label: 'Home' },
  { icon: IconGauge, label: 'Dashboard' },
  { icon: IconDeviceDesktopAnalytics, label: 'Analytics' },
  { icon: IconCalendarStats, label: 'Releases' },
  { icon: IconUser, label: 'Account' },
  { icon: IconFingerprint, label: 'Security' },
  { icon: IconSettings, label: 'Settings' },
];

const linksMockdata = [
  'Security',
  'Settings',
  'Dashboard',
  'Releases',
  'Account',
  'Orders',
  'Clients',
  'Databases',
  'Pull Requests',
  'Open Issues',
  'Wiki pages',
];

export default function DoubleNavbar() {
  const [active, setActive] = useState('Releases');
  const [activeLink, setActiveLink] = useState('Settings');

  return (
    <nav className="bg-[rgb(10,10,10)] h-screen w-[300px] flex flex-col border-r border-gray-700">
      <div className="flex flex-1">
        {/* Left sidebar with icons */}
        <div className="flex-none w-[60px] bg-[rgb(10,10,10)] flex flex-col items-center border-r border-gray-700">
          {/* Logo */}
          <div className="w-full flex justify-center h-[60px] pt-4 border-b border-gray-700 mb-6">
            <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center text-white font-bold">
              ?
            </div>
          </div>
          
          {/* Main navigation icons */}
          {mainLinksMockdata.map((link) => (
            <button
              key={link.label}
              onClick={() => setActive(link.label)}
              className={`w-11 h-11 mb-2 flex items-center justify-center rounded-md transition-colors ${
                active === link.label
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-400 hover:bg-gray-700 hover:text-white'
              }`}
              title={link.label}
            >
              <link.icon size={22} stroke={1.5} />
            </button>
          ))}
        </div>
        
        {/* Right sidebar with text links */}
        <div className="flex-1 bg-[rgb(20,20,20)]">
          {/* Title */}
          <h4 className="font-medium text-lg text-white bg-[rgb(10,10,10)] px-4 py-[18px] h-[60px] border-b border-gray-700 mb-6">
            {active}
          </h4>
          
          {/* Links */}
          <div className="px-2">
            {linksMockdata.map((link) => (
              <Link
                href="#"
                key={link}
                onClick={(event) => {
                  event.preventDefault();
                  setActiveLink(link);
                }}
                className={`block text-sm font-medium h-11 leading-[44px] px-4 rounded-r-md transition-colors ${
                  activeLink === link
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                }`}
              >
                {link}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
