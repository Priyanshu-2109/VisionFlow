'use client';

import { useState } from 'react';
import Image from 'next/image';
import { FaGoogle, FaFacebook } from 'react-icons/fa';
import Link from 'next/link';
import { FaArrowLeft } from "react-icons/fa";
import { useRouter } from "next/navigation"; // For navigation
export default function SignIn() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const router = useRouter(); // Next.js router for back navigation

  return (
    <div className="flex h-screen bg">
      {/* Left Side */}
      {/* <div className="hidden md:flex w-1/2 bg-gray-400 p-10 flex-col justify-between">
        <div>
          <button className=' bg-gray-700 p-3 rounded-xl'>
            <Link href={"/"}>back to HOME</Link>
          </button>
        </div>
        <div className="text-3xl font-semibold text-gray-700">
          Multipurpose tool to succeed your business
        </div>
        <div></div>
      </div> */}

      <div
        className="hidden md:flex w-1/2 bg-cover bg-center bg-no-repeat p-10 flex-col justify-between"
        style={{ backgroundImage: "url('/log.jpg')" }} // Change path accordingly
      >
        <div>
          <button className="text-white text-2xl">
            <Link href={"/"}><FaArrowLeft /></Link>
          </button>
        </div>
        <div className="text-3xl font-semibold flex ">
          <span className=' font-bold p-2 rounded-sm'>Multipurpose tool to succeed your business</span>
        </div>
        <div></div>
      </div>

      {/* Right Side */}
      <div className="w-full md:w-1/2 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          <h2 className="text-2xl font-bold mb-6">Sign in</h2>

          <div className="space-y-4">
            <input
              type="email"
              placeholder="Email address"
              className="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-indigo-500"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <input
              type="password"
              placeholder="Password"
              className="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-indigo-500"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <div className="text-right mt-2 text-sm text-indigo-500 cursor-pointer">
            Forgot password?
          </div>

          <button className="w-full bg-indigo-600 text-white py-3 rounded-lg mt-4 hover:bg-indigo-700">
            Sign In
          </button>

          <div className="flex items-center justify-center my-4">
            <span className="w-1/3 border-b"></span>
            <span className="mx-2 text-sm text-gray-500">or</span>
            <span className="w-1/3 border-b"></span>
          </div>

          <div className="flex space-x-4">
            <button className="w-1/2 flex items-center justify-center border rounded-lg py-3 hover:bg-gray-700">
              <FaGoogle className="mr-2" /> Google
            </button>
            <button className="w-1/2 flex items-center justify-center border rounded-lg py-3 hover:bg-gray-700">
              <FaFacebook className="mr-2 text-blue-600" /> Facebook
            </button>
          </div>

          <p className="mt-4 text-sm text-center text-gray-600">
            Don't have an account? <span className="text-indigo-500 cursor-pointer">
              <Link href={"/signup"}>
                Sign Up
              </Link>
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}
