'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();
  const [roomId, setRoomId] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const startNewChat = () => {
    setIsLoading(true);
    // Generate a unique room ID
    const newRoomId = `room-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    router.push(`/chat/${newRoomId}`);
  };

  const joinExistingRoom = () => {
    if (roomId.trim()) {
      setIsLoading(true);
      router.push(`/chat/${roomId.trim()}`);
    } else {
      alert('กรุณาใส่ Room ID');
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="max-w-md mx-auto bg-white rounded-2xl shadow-xl p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl text-white">🩺</span>
            </div>
            <h1 className="text-2xl font-bold text-gray-800 mb-2">
              ระบบปรึกษาแพทย์ออนไลน์
            </h1>
            <p className="text-gray-600 text-sm">
              ปรึกษาอาการเบื้องต้นกับแพทย์ผู้เชี่ยวชาญ
            </p>
          </div>

          {/* New Chat Button */}
          <div className="mb-6">
            <button
              onClick={startNewChat}
              disabled={isLoading}
              className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white font-medium py-4 px-6 rounded-xl transition-colors flex items-center justify-center space-x-2"
            >
              {isLoading ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              ) : (
                <>
                  <span>💬</span>
                  <span>เริ่มการปรึกษาใหม่</span>
                </>
              )}
            </button>
          </div>

          {/* Divider */}
          <div className="flex items-center mb-6">
            <div className="flex-1 border-t border-gray-200"></div>
            <span className="px-4 text-sm text-gray-500">หรือ</span>
            <div className="flex-1 border-t border-gray-200"></div>
          </div>

          {/* Join Existing Room */}
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Room ID (สำหรับกลับมาสนทนาต่อ)
              </label>
              <input
                type="text"
                value={roomId}
                onChange={(e) => setRoomId(e.target.value)}
                placeholder="room-xxxxx-xxxxx"
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                onKeyPress={(e) => e.key === 'Enter' && joinExistingRoom()}
              />
            </div>
            <button
              onClick={joinExistingRoom}
              disabled={isLoading || !roomId.trim()}
              className="w-full bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 text-gray-700 font-medium py-3 px-6 rounded-xl transition-colors"
            >
              🔗 เข้าร่วมห้องสนทนา
            </button>
          </div>

          {/* Info */}
          <div className="mt-8 p-4 bg-yellow-50 rounded-xl border border-yellow-200">
            <div className="flex items-start space-x-2">
              <span className="text-yellow-600 mt-0.5">⚠️</span>
              <div className="text-sm text-yellow-800">
                <p className="font-medium mb-1">ข้อมูลสำคัญ:</p>
                <ul className="space-y-1 text-xs">
                  <li>• บันทึก Room ID เพื่อกลับมาสนทนาต่อได้</li>
                  <li>• ข้อมูลเป็นคำแนะนำเบื้องต้นเท่านั้น</li>
                  <li>• หากมีอาการรุนแรง กรุณาพบแพทย์ทันที</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Doctor Dashboard Link */}
          <div className="mt-6 text-center">
            <a
              href="/doctor"
              className="text-sm text-blue-600 hover:text-blue-800 underline"
            >
              สำหรับแพทย์ → เข้าสู่ระบบแพทย์
            </a>
          </div>
        </div>
      </div>
    </main>
  );
}
