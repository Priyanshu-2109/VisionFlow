"use client"
import Link from 'next/link';
import React from 'react';
import { FaGoogle, FaFacebook } from 'react-icons/fa';
import { FaArrowLeft } from "react-icons/fa";
import { useRouter } from "next/navigation"; // For navigation

const Register = () => {
    const router = useRouter(); // Next.js router for back navigation

    return (
        <>
            <div className=' pt-20 px-24'>
                <button onClick={() => router.back()} className="text-white text-2xl">
                    <FaArrowLeft />
                </button>
            </div>
            <div className="flex items-center justify-center">
                <div className=" p-8 rounded shadow-md w-full max-w-md">
                    <h2 className="text-2xl font-bold mb-6 text-center">Welcome to Version Flow</h2>
                    <form>
                        <div className="mb-4">
                            <label className="block text-gray-500">Email address</label>
                            <input
                                type="email"
                                className="w-full p-2 border border-gray-300 rounded mt-1"
                                placeholder="Email address"
                            />
                        </div>
                        <div className="mb-6">
                            <label className="block text-gray-500">Password</label>
                            <input
                                type="password"
                                className="w-full p-2 border border-gray-300 rounded mt-1"
                                placeholder="Password"
                            />
                        </div>
                        <button className="w-full bg-purple-600 text-white p-2 rounded hover:bg-purple-700">
                            Register
                        </button>
                    </form>
                    <div className="mt-6 text-center">
                        <div className="flex items-center justify-center my-4">
                            <span className="w-1/3 border-b"></span>
                            <span className="mx-2 text-sm text-gray-500">or sign with</span>
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
                    </div>
                </div>
            </div>
        </>
    );
};

export default Register;
